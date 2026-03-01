"""Ray-based multi-worker training loop utilities.

This module contains learner-side orchestration code for collecting rollouts
from Ray workers and applying updates on the central learner.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import math
import time
import numpy as np

from rllib.model_free.common.utils.train_utils import _maybe_call, _infer_env_id, _wrap_make_env_with_normalize, _reset_pbar

_AGENT_STALL_SEC = 30.0


def train_ray(trainer: Any, pbar: Any, msg_pbar: Any) -> None:
    """
    Ray multi-worker rollout loop (learner-side).

    This loop collects rollout chunks from Ray workers, ingests them into the learner
    algorithm (`trainer.algo`), runs learner updates, and periodically synchronizes
    policy weights (and optionally NormalizeWrapper running statistics) back to workers.

    High-level flow
    ---------------
    Setup:
      1) Ensure `trainer.ray_env_make_fn` exists (auto-infer from train_env if possible).
      2) Ensure `trainer.ray_policy_spec` exists (from algo factory spec if possible).
      3) Construct a Ray runner (RayLearner) and broadcast initial weights/env state.

    Main loop:
      A) (Off-policy only) Optionally pause rollout after warmup to run updates-only.
      B) Collect rollout from workers.
      C) Ingest rollout into algo.
      D) Fire per-transition callbacks (on_step).
      E) Run updates (on-policy or off-policy mode).
      F) Periodically broadcast weights and env normalization stats to workers.
      G) Periodically log system counters and update progress bar postfix.

    Parameters
    ----------
    trainer : Any
        Trainer-like object (duck-typed) providing:
          - algo : learner algorithm
              - is_off_policy : bool (optional)
              - ready_to_update() -> bool
              - update() -> Mapping[str, Any] or {}
              - on_env_step(tr) or on_rollout(rollout) (optional)
              - (on-policy only) remaining_rollout_steps() -> int
              - (off-policy only) warmup_steps / update_after and internal _env_steps (optional)
          - train_env : environment (used for normalization state export)
          - n_envs : int
          - rollout_steps_per_env : int
          - seed : int
          - utd : float (used to scale update budget)
          - total_env_steps : int (stop condition; counts learner-ingested transitions)
          - global_env_step : int (incremented by ingested transitions)
          - global_update_step : int (incremented by inferred update count)
          - episode_idx : int (sys logging only; in Ray mode this is often approximate)
          - log_every_steps : int (sys logging cadence; <=0 disables)
          - _normalize_enabled : bool
          - ray_env_make_fn : Optional[Callable[[], Any]]
          - ray_policy_spec : Optional[Any] (PolicyFactorySpec)
          - ray_want_onpolicy_extras : Optional[bool]
          - sync_weights_every_updates : int (<=0 disables)
          - pause_rollout_after_warmup : bool (off-policy only, optional; default True)
          - max_episode_steps : Optional[int]
          - flatten_obs : bool
          - step_obs_dtype : dtype
          - callbacks : optional (on_step/on_update)
          - logger : optional (log(metrics, step, prefix, pbar=...))
          - _stop_training : bool flag (optional; if True breaks)
          - _warn(msg) : optional warning hook

    pbar : Any
        Progress bar-like object supporting:
          - update(int)
          - set_postfix(dict, refresh=...)
    msg_pbar : Any
        A progress-bar-like object to pass into logger calls.

    Returns
    -------
    None

    Notes
    -----
    On-policy vs off-policy update behavior
    --------------------------------------
    - On-policy: expects `algo.remaining_rollout_steps()`; collection is trimmed to exactly
      the remaining batch size. When ready, performs exactly one update per iteration.
    - Off-policy: performs up to `max_update_calls_per_iter` update() calls per iteration
      while `algo.ready_to_update()` is True.

    Ray initialization
    ------------------
    Ray is initialized best-effort if not already initialized (inside `_build_ray_runner`).
    """
    _ensure_ray_env_make_fn(trainer)
    _ensure_ray_policy_spec(trainer)

    algo = getattr(trainer, "algo", None)
    is_off_policy = bool(getattr(algo, "is_off_policy", False))
    want_onpolicy_extras = _infer_want_onpolicy_extras(trainer, is_off_policy=is_off_policy)

    runner, get_policy_state_dict_cpu_fn = _build_ray_runner(
        trainer,
        want_onpolicy_extras=want_onpolicy_extras,
    )

    _initial_broadcast(trainer, runner, get_policy_state_dict_cpu_fn)

    last_sys_log_step = int(getattr(trainer, "global_env_step", 0))
    max_update_calls_per_iter = _infer_max_update_calls_per_iter(trainer)
    agent_pbars = getattr(trainer, "_agent_pbars", None)
    agent_pbar_map = getattr(trainer, "_agent_pbar_map", None)
    if agent_pbar_map is None:
        agent_pbar_map = {}
        try:
            trainer._agent_pbar_map = agent_pbar_map
        except Exception:
            pass

    total_env_steps = int(getattr(trainer, "total_env_steps", 0))
    while int(getattr(trainer, "global_env_step", 0)) < total_env_steps:
        # ------------------------------------------------------------------
        # (A) Off-policy: optional "updates-only" phase after warmup
        # ------------------------------------------------------------------
        if is_off_policy:
            warmup = int(getattr(algo, "warmup_steps", 0))
            update_after = int(getattr(algo, "update_after", 0))
            gate = max(warmup, update_after)

            # In your OffPolicyAlgorithm, `_env_steps` is incremented in on_env_step.
            env_steps = int(getattr(algo, "_env_steps", 0))
            pause_after = bool(getattr(trainer, "pause_rollout_after_warmup", True))

            if pause_after and env_steps >= gate and algo.ready_to_update():
                if not run_updates(trainer, mode="offpolicy", max_calls=max_update_calls_per_iter, pbar=msg_pbar):
                    return

                maybe_sync_worker_weights_and_env(trainer, runner, get_policy_state_dict_cpu_fn)
                last_sys_log_step = _maybe_log_sys_ray(trainer, last_sys_log_step, pbar=pbar)
                if getattr(trainer, "_stop_training", False):
                    break
                continue

        # ------------------------------------------------------------------
        # (B) Collect rollout from workers
        # ------------------------------------------------------------------
        per_worker, target_new = _decide_collection_size(trainer, is_off_policy=is_off_policy)
        try:
            trainer._ray_per_worker_target = int(per_worker)
        except Exception:
            pass
        base_step_collect = int(getattr(trainer, "global_env_step", 0))
        window_start_ts = float(time.time())
        agent_step_counts: Dict[str, int] = {f"agent_{i:02d}": 0 for i in range(int(getattr(trainer, "n_envs", 0)))}
        agent_last_progress_ts: Dict[str, float] = {k: window_start_ts for k in agent_step_counts.keys()}
        try:
            trainer._ray_agent_step_counts_window = dict(agent_step_counts)
            trainer._ray_agent_last_progress_ts = dict(agent_last_progress_ts)
        except Exception:
            pass
        last_render_ts = [0.0]

        # Reset per-agent pbars to show *this* collection window only.
        if agent_pbars:
            for pb in agent_pbars:
                _reset_pbar(pb, int(per_worker))

        streaming = callable(getattr(runner, "collect_stream", None))
        n_new = 0
        rollout = None
        if streaming:
            def _progress_cb(chunk: List[Dict[str, Any]]) -> None:
                """Handle one streamed rollout chunk from workers.

                Parameters
                ----------
                chunk : List[Dict[str, Any]]
                    Transition chunk returned by ``RayLearner.collect_stream``.
                """
                nonlocal n_new
                n_chunk = int(len(chunk))
                if n_chunk <= 0:
                    return
                n_new += n_chunk
                _pbar_update_best_effort(pbar, n_chunk)
                now_ts = time.time()

                # Update per-agent progress bars (best-effort).
                counts: Dict[str, int] = {}
                for tr in chunk:
                    if not isinstance(tr, Mapping):
                        continue
                    env_id = tr.get("env_id")
                    info = tr.get("info")
                    if env_id is None and isinstance(info, Mapping):
                        env_id = info.get("env_id")
                    if env_id is None:
                        continue
                    env_id = str(env_id)
                    counts[env_id] = counts.get(env_id, 0) + 1
                    agent_step_counts[env_id] = agent_step_counts.get(env_id, 0) + 1
                    agent_last_progress_ts[env_id] = now_ts
                try:
                    trainer._ray_agent_step_counts_window = dict(agent_step_counts)
                    trainer._ray_agent_last_progress_ts = dict(agent_last_progress_ts)
                except Exception:
                    pass

                if agent_pbars:
                    for env_id in counts.keys():
                        if env_id in agent_pbar_map:
                            continue
                        if len(agent_pbar_map) >= len(agent_pbars):
                            break
                        idx = len(agent_pbar_map)
                        agent_pbar_map[env_id] = idx
                        try:
                            agent_pbars[idx].set_description_str(f"Agent_{idx + 1:02d}", refresh=True)
                        except Exception:
                            pass
                    for env_id, n in counts.items():
                        idx = agent_pbar_map.get(env_id)
                        if idx is None:
                            continue
                        try:
                            agent_pbars[idx].update(int(n))
                        except Exception:
                            pass

                if (now_ts - float(last_render_ts[0])) >= 0.5:
                    _render_text_dashboard(
                        trainer,
                        per_worker=int(per_worker),
                        agent_step_counts=agent_step_counts,
                        agent_last_progress_ts=agent_last_progress_ts,
                        global_step=int(base_step_collect + n_new),
                        window_elapsed_sec=max(0.0, float(time.time()) - window_start_ts),
                        final=False,
                    )
                    last_render_ts[0] = now_ts

                # Update global episode counters/postfix (best-effort).
                try:
                    for tr in chunk:
                        if not isinstance(tr, Mapping):
                            continue
                        info = tr.get("info")
                        if not (isinstance(info, Mapping) and bool(tr.get("done", False))):
                            continue
                        if "episode_return" in info:
                            trainer.record_episode_stats(float(info.get("episode_return", 0.0)))
                    _set_global_pbar_postfix(trainer, pbar)
                except Exception:
                    pass

            rollout = runner.collect_stream(
                deterministic=False,
                n_steps=int(per_worker),
                progress_chunk=int(getattr(trainer, "ray_progress_chunk", 1)),
                progress_cb=_progress_cb,
            )
        else:
            rollout = runner.collect(deterministic=False, n_steps=int(per_worker))
            n_new = int(len(rollout))

        n_total = int(len(rollout))
        if n_total <= 0:
            continue

        rollout, n_new = _trim_onpolicy_rollout(rollout, target_new=target_new)
        if n_new <= 0:
            # on-policy case where remaining==0 may yield empty trimmed rollout
            # (outer loop will run updates and continue).
            pass

        # ------------------------------------------------------------------
        # (C) Bookkeeping and progress bar
        # ------------------------------------------------------------------
        base_step = int(getattr(trainer, "global_env_step", 0))
        trainer.global_env_step = base_step + int(n_new)
        _render_text_dashboard(
            trainer,
            per_worker=int(per_worker),
            agent_step_counts=agent_step_counts,
            agent_last_progress_ts=agent_last_progress_ts,
            global_step=int(getattr(trainer, "global_env_step", 0)),
            window_elapsed_sec=max(0.0, float(time.time()) - window_start_ts),
            final=True,
        )
        try:
            trainer._ray_agent_step_counts_window = dict(agent_step_counts)
            trainer._ray_agent_last_progress_ts = dict(agent_last_progress_ts)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # (D) Ingest rollout into algo
        # ------------------------------------------------------------------
        _ingest_rollout(trainer, rollout)

        # Per-transition logging (separate from metrics)
        try:
            for i, tr in enumerate(rollout):
                env_id = None
                info = tr.get("info") if isinstance(tr, Mapping) else None
                if isinstance(tr, Mapping) and ("env_id" in tr):
                    env_id = tr.get("env_id")
                if env_id is None and isinstance(info, Mapping) and ("env_id" in info):
                    env_id = info.get("env_id")
                trainer.log_transition(tr, timestep=int(base_step + i + 1), env_id=env_id)

                if isinstance(tr, Mapping) and bool(tr.get("done", False)) and isinstance(info, Mapping):
                    if "episode_return" in info and "episode_len" in info:
                        trainer.log_agent_episode_end(
                            episode_return=float(info.get("episode_return", 0.0)),
                            episode_len=int(info.get("episode_len", 0)),
                            env_id=env_id,
                            timestep=int(base_step + i + 1),
                        )
        except Exception:
            pass

        # Per-transition callbacks (preserve existing behavior)
        if getattr(trainer, "callbacks", None) is not None:
            for tr in rollout:
                cont = _maybe_call(trainer.callbacks, "on_step", trainer, transition=tr)
                if cont is False:
                    return

        # ------------------------------------------------------------------
        # (E) Learner updates
        # ------------------------------------------------------------------
        if is_off_policy:
            ok = run_updates(trainer, mode="offpolicy", max_calls=max_update_calls_per_iter, pbar=msg_pbar)
        else:
            ok = run_updates(trainer, mode="onpolicy", max_calls=1, pbar=msg_pbar)
        if not ok:
            return

        # ------------------------------------------------------------------
        # (F) Sync weights (and optional env stats)
        # ------------------------------------------------------------------
        maybe_sync_worker_weights_and_env(trainer, runner, get_policy_state_dict_cpu_fn)

        # ------------------------------------------------------------------
        # (G) Sys log + UI
        # ------------------------------------------------------------------
        last_sys_log_step = _maybe_log_sys_ray(trainer, last_sys_log_step, pbar=pbar)
        _maybe_set_updates_postfix(trainer, pbar)

        if getattr(trainer, "_stop_training", False):
            break


# =============================================================================
# Updates
# =============================================================================
def _infer_num_updates_from_metrics(metrics: Any) -> int:
    """
    Infer how many learner updates were performed from an update metrics mapping.

    Parameters
    ----------
    metrics : Any
        Typically the return value of `algo.update()`. If mapping-like, we attempt
        to read a conventional key that indicates how many updates were executed
        by that call.

    Returns
    -------
    n_updates : int
        Positive integer update increment (>= 1).

    Convention
    ----------
    Preference order (first present and parseable wins):
      - "sys/num_updates"
      - "num_updates"
      - "n_updates"
      - "update_steps"
      - "train/num_updates"

    Notes
    -----
    If no suitable key is found or parsing fails, defaults to 1 meaning:
      one `update()` call counts as one learner update step.
    """
    if not isinstance(metrics, Mapping):
        return 1

    for key in ("sys/num_updates", "num_updates", "n_updates", "update_steps", "train/num_updates"):
        v = metrics.get(key, None)
        if v is None:
            continue
        try:
            k = int(v)
            return max(1, k)
        except Exception:
            pass
    return 1


def run_updates(trainer: Any, *, mode: str, max_calls: int, pbar: Any) -> bool:
    """
    Run learner updates in either on-policy or off-policy mode.

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing:
          - algo.ready_to_update() -> bool
          - algo.update() -> metrics mapping (or {})
          - global_update_step : int (incremented here)
          - callbacks.on_update(...) (optional)
          - logger.log(...) (optional)
    mode : {"onpolicy", "offpolicy"}
        Update regime:
          - "onpolicy": at most one update() call when ready_to_update() is True.
          - "offpolicy": repeated update() calls while ready_to_update() and budget allows.
    max_calls : int
        Maximum number of `algo.update()` calls to issue (off-policy only).
    pbar : Any
        Progress bar-like object passed into logger (optional).

    Returns
    -------
    ok : bool
        False if callbacks request stop (on_update returns False), else True.

    Notes
    -----
    - Each update() call increments trainer.global_update_step by k where k is
      inferred from metrics via `_infer_num_updates_from_metrics`.
    - Off-policy path increments `n_calls` per update() call to avoid infinite loops.
    """
    if mode not in ("onpolicy", "offpolicy"):
        raise ValueError(f"Unknown update mode: {mode}")

    algo = getattr(trainer, "algo", None)

    if mode == "onpolicy":
        if not algo.ready_to_update():
            return True

        metrics = algo.update()
        k = _infer_num_updates_from_metrics(metrics)
        trainer.global_update_step = int(getattr(trainer, "global_update_step", 0)) + k

        return _handle_update_side_effects(trainer, metrics, pbar=pbar)

    # off-policy
    n_calls = 0
    while n_calls < int(max_calls):
        if not algo.ready_to_update():
            break

        metrics = algo.update()
        k = _infer_num_updates_from_metrics(metrics)
        trainer.global_update_step = int(getattr(trainer, "global_update_step", 0)) + k

        n_calls += 1  # important: ensures loop control progresses

        if not _handle_update_side_effects(trainer, metrics, pbar=pbar):
            return False

    return True


# =============================================================================
# Worker synchronization
# =============================================================================
def maybe_sync_worker_weights_and_env(
    trainer: Any,
    runner: Any,
    get_policy_state_dict_cpu_fn: Callable[[Any], Dict[str, Any]],
) -> bool:
    """
    Broadcast latest policy weights (and optionally NormalizeWrapper state) to Ray workers.

    Parameters
    ----------
    trainer : Any
        Trainer-like object that may provide:
          - sync_weights_every_updates : int
          - global_update_step : int
          - algo : policy holder
          - train_env : optional, may implement state_dict()
          - _normalize_enabled : bool
    runner : Any
        Ray runner providing:
          - broadcast_policy(state_dict)
          - broadcast_env_state(env_state)
    get_policy_state_dict_cpu_fn : Callable[[Any], Dict[str, Any]]
        Function that returns a CPU (device-agnostic) state_dict suitable for broadcast.

    Returns
    -------
    did_sync : bool
        True if synchronization was performed; False otherwise.

    Notes
    -----
    Sync condition:
      - `sync_weights_every_updates` must be > 0, and
      - `global_update_step % sync_weights_every_updates == 0`

    Normalization state sync:
      - If `_normalize_enabled` is True and `train_env.state_dict()` exists,
        broadcast that env state as well (best-effort).
    """
    every = int(getattr(trainer, "sync_weights_every_updates", 0))
    if every <= 0:
        return False

    if (int(getattr(trainer, "global_update_step", 0)) % every) != 0:
        return False

    runner.broadcast_policy(get_policy_state_dict_cpu_fn(getattr(trainer, "algo", None)))

    train_env = getattr(trainer, "train_env", None)
    if bool(getattr(trainer, "_normalize_enabled", False)) and callable(getattr(train_env, "state_dict", None)):
        try:
            runner.broadcast_env_state(train_env.state_dict())
        except Exception:
            pass

    return True


# =============================================================================
# Internal helpers
# =============================================================================
def _pbar_update_best_effort(pbar: Any, n: int) -> None:
    """Best-effort progress bar update."""
    try:
        pbar.update(int(n))
    except Exception:
        pass


def _maybe_set_updates_postfix(trainer: Any, pbar: Any) -> None:
    """Best-effort progress-bar postfix update for global update count.

    Parameters
    ----------
    trainer : Any
        Trainer-like object exposing ``global_update_step``.
    pbar : Any
        Progress bar object supporting ``set_postfix``.

    Returns
    -------
    None

    Notes
    -----
    This helper is intentionally resilient: exceptions are swallowed so pbar
    rendering errors never interrupt training.
    """
    try:
        pbar.set_postfix({"updates": int(getattr(trainer, "global_update_step", 0))}, refresh=False)
    except Exception:
        pass


def _ensure_ray_env_make_fn(trainer: Any) -> None:
    """
    Ensure `trainer.ray_env_make_fn` is populated.

    If missing, attempts to infer an env_id from `trainer.train_env` and creates
    a default `gym.make(env_id)` factory. If normalization is enabled, wraps the
    factory with NormalizeWrapper parameters recovered from `trainer.train_env`.

    Raises
    ------
    ValueError
        If env_id cannot be inferred and no ray_env_make_fn is provided.
    RuntimeError
        If normalization is enabled but required normalization parameters cannot
        be recovered from train_env (e.g., obs_shape).
    """
    if getattr(trainer, "ray_env_make_fn", None) is not None:
        return

    env_id = _infer_env_id(getattr(trainer, "train_env", None))
    if env_id is None:
        raise ValueError(
            "n_envs > 1 requires ray_env_make_fn OR a resolvable train_env.spec.id to auto-make envs."
        )

    def _default_make_env() -> Any:
        """Create one default environment instance from inferred ``env_id``.

        Returns
        -------
        Any
            Fresh environment instance from ``gym.make(env_id)``.
        """
        try:
            import gymnasium as gym
        except Exception:
            import gym  # type: ignore
        return gym.make(env_id)

    make_fn: Callable[[], Any] = _default_make_env

    if bool(getattr(trainer, "_normalize_enabled", False)):
        te = getattr(trainer, "train_env", None)
        obs_shape = getattr(te, "obs_shape", None)
        if obs_shape is None:
            raise RuntimeError("normalize=True but could not recover obs_shape from train_env.")

        make_fn = _wrap_make_env_with_normalize(
            make_fn,
            obs_shape=tuple(obs_shape),
            norm_obs=bool(getattr(te, "norm_obs", True)),
            norm_reward=bool(getattr(te, "norm_reward", False)),
            clip_obs=float(getattr(te, "clip_obs", 10.0)),
            clip_reward=float(getattr(te, "clip_reward", 10.0)),
            gamma=float(getattr(te, "gamma", 0.99)),
            epsilon=float(getattr(te, "epsilon", 1e-8)),
            training=True,
            max_episode_steps=getattr(trainer, "max_episode_steps", None),
            action_rescale=bool(getattr(te, "action_rescale", False)),
            clip_action=float(getattr(te, "clip_action", 0.0)),
            reset_return_on_done=bool(getattr(te, "reset_return_on_done", True)),
            reset_return_on_trunc=bool(getattr(te, "reset_return_on_trunc", True)),
            obs_dtype=getattr(te, "obs_dtype", np.float32),
        )

    trainer.ray_env_make_fn = make_fn


def _ensure_ray_policy_spec(trainer: Any) -> None:
    """
    Ensure `trainer.ray_policy_spec` is populated.

    If missing, uses `trainer.algo.get_ray_policy_factory_spec()`.

    Raises
    ------
    ValueError
        If neither trainer.ray_policy_spec nor algo.get_ray_policy_factory_spec is available.
    """
    if getattr(trainer, "ray_policy_spec", None) is not None:
        return

    fn = getattr(getattr(trainer, "algo", None), "get_ray_policy_factory_spec", None)
    if not callable(fn):
        raise ValueError("n_envs > 1 requires ray_policy_spec OR algo.get_ray_policy_factory_spec().")
    trainer.ray_policy_spec = fn()


def _infer_want_onpolicy_extras(trainer: Any, *, is_off_policy: bool) -> bool:
    """
    Infer whether Ray workers should return on-policy extras (logp, value, etc.).

    Parameters
    ----------
    trainer : Any
        Trainer-like object possibly providing `ray_want_onpolicy_extras`.
    is_off_policy : bool
        Whether the learner algorithm is off-policy.

    Returns
    -------
    want : bool
        Whether workers should request/return on-policy extras.

    Precedence
    ----------
    1) If trainer.ray_want_onpolicy_extras is not None: use it.
    2) Else: True for on-policy algorithms, False for off-policy algorithms.
    """
    if getattr(trainer, "ray_want_onpolicy_extras", None) is not None:
        return bool(trainer.ray_want_onpolicy_extras)
    return (not bool(is_off_policy))


def _build_ray_runner(trainer: Any, *, want_onpolicy_extras: bool) -> Tuple[Any, Callable[[Any], Dict[str, Any]]]:
    """
    Construct the Ray runner (RayLearner) and return it along with a policy state extractor.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with ray config fields.
    want_onpolicy_extras : bool
        Whether workers should request on-policy extras.

    Returns
    -------
    runner : Any
        Instance of RayLearner.
    get_policy_state_dict_cpu_fn : Callable[[Any], Dict[str, Any]]
        Function that converts a policy/algo into a CPU state_dict for broadcasting.

    Notes
    -----
    Imports Ray lazily to keep this module importable without Ray.
    """
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    from rllib.model_free.common.trainers.ray_workers import RayLearner
    from rllib.model_free.common.utils.ray_utils import _get_policy_state_dict_cpu

    runner = RayLearner(
        env_make_fn=getattr(trainer, "ray_env_make_fn", None),
        policy_spec=getattr(trainer, "ray_policy_spec", None),
        n_workers=int(getattr(trainer, "n_envs", 1)),
        steps_per_worker=int(getattr(trainer, "rollout_steps_per_env", 1)),
        base_seed=int(getattr(trainer, "seed", 0)),
        max_episode_steps=getattr(trainer, "max_episode_steps", None),
        want_onpolicy_extras=bool(want_onpolicy_extras),
        flatten_obs=bool(getattr(trainer, "flatten_obs", False)),
        obs_dtype=getattr(trainer, "step_obs_dtype", np.float32),
    )
    return runner, _get_policy_state_dict_cpu


def _initial_broadcast(
    trainer: Any,
    runner: Any,
    get_policy_state_dict_cpu_fn: Callable[[Any], Dict[str, Any]],
) -> None:
    """
    Initial broadcast of policy weights and (optional) env normalization state to workers.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with `algo`, `train_env`, and normalization flags.
    runner : Any
        Ray runner exposing broadcast_policy / broadcast_env_state.
    get_policy_state_dict_cpu_fn : Callable
        Policy state_dict CPU extractor.
    """
    runner.broadcast_policy(get_policy_state_dict_cpu_fn(getattr(trainer, "algo", None)))

    train_env = getattr(trainer, "train_env", None)
    if bool(getattr(trainer, "_normalize_enabled", False)) and callable(getattr(train_env, "state_dict", None)):
        try:
            runner.broadcast_env_state(train_env.state_dict())
        except Exception:
            pass


def _infer_max_update_calls_per_iter(trainer: Any) -> int:
    """
    Infer off-policy update-call budget per iteration.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with `utd` configuration.

    Returns
    -------
    max_calls : int
        Maximum number of update() calls allowed per iteration (>= 1).

    Notes
    -----
    Current policy:
      - base budget = 10_000 calls
      - scaled by utd when utd > 0
    This is intentionally generous; the `ready_to_update()` gate usually limits it.
    """
    base = 10_000
    utd = float(getattr(trainer, "utd", 1.0))
    max_calls = int(base * utd) if utd > 0.0 else base
    return max(1, int(max_calls))


def _decide_collection_size(trainer: Any, *, is_off_policy: bool) -> Tuple[int, Optional[int]]:
    """
    Decide how many steps each worker should collect this iteration.

    Parameters
    ----------
    trainer : Any
        Trainer-like object containing:
          - n_envs
          - rollout_steps_per_env
          - algo.remaining_rollout_steps() (on-policy only)
    is_off_policy : bool
        Whether learner algorithm is off-policy.

    Returns
    -------
    per_worker : int
        Number of steps per worker to collect this iteration.
    target_new : int or None
        For on-policy, total desired number of new transitions to ingest. The
        collected rollout may be trimmed to exactly this size. For off-policy, None.

    Raises
    ------
    ValueError
        If on-policy mode is requested but algo.remaining_rollout_steps() is missing.
    """
    n_envs = int(getattr(trainer, "n_envs", 1))
    steps_per_env = int(getattr(trainer, "rollout_steps_per_env", 1))

    if is_off_policy:
        return steps_per_env, None

    fn_rem = getattr(getattr(trainer, "algo", None), "remaining_rollout_steps", None)
    if not callable(fn_rem):
        raise ValueError("Ray on-policy path requires algo.remaining_rollout_steps().")

    remaining = int(fn_rem())
    if remaining <= 0:
        # Return a tiny collection; outer loop will then perform updates.
        return 1, 0

    per_worker = int(math.ceil(float(remaining) / float(n_envs)))
    per_worker = max(1, min(per_worker, steps_per_env))
    return per_worker, int(remaining)


def _trim_onpolicy_rollout(rollout: Any, *, target_new: Optional[int]) -> Tuple[Any, int]:
    """
    Trim rollout to at most `target_new` transitions (on-policy only).

    Parameters
    ----------
    rollout : sequence
        Collected rollout transitions.
    target_new : int or None
        Desired total transition count. If None, no trimming is applied.

    Returns
    -------
    rollout_out : same type as input
        Possibly sliced rollout.
    n_new : int
        Number of transitions after trimming.
    """
    n_total = int(len(rollout))
    if target_new is None:
        return rollout, n_total

    if int(target_new) <= 0:
        return rollout[:0], 0

    if n_total > int(target_new):
        return rollout[: int(target_new)], int(target_new)

    return rollout, n_total


def _ingest_rollout(trainer: Any, rollout: Any) -> None:
    """
    Feed collected rollout into algo.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with `algo`.
    rollout : sequence
        Transition sequence.

    Notes
    -----
    Prefers `algo.on_rollout(rollout)` if available, else falls back to calling
    `algo.on_env_step(tr)` for each transition.
    """
    algo = getattr(trainer, "algo", None)

    on_rollout = getattr(algo, "on_rollout", None)
    if callable(on_rollout):
        on_rollout(rollout)
        return

    for tr in rollout:
        algo.on_env_step(tr)


def _handle_update_side_effects(trainer: Any, metrics: Any, pbar: Any) -> bool:
    """
    Apply side effects after a single algo.update() call.

    Side effects
    ------------
    - callbacks.on_update(trainer, metrics=...)
    - logger.log(metrics, step=global_env_step, prefix="train/", pbar=pbar)

    Parameters
    ----------
    trainer : Any
        Trainer-like object with callbacks/logger.
    metrics : Any
        Update metrics returned by algo.update().
    pbar : Any
        Progress bar-like object passed into logger.

    Returns
    -------
    ok : bool
        False if callbacks request stop (on_update returns False), else True.
    """
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else None
    if metrics_map:
        try:
            trainer.log_agent_update(metrics_map)
        except Exception:
            pass

    if getattr(trainer, "callbacks", None) is not None:
        cont = _maybe_call(trainer.callbacks, "on_update", trainer, metrics=metrics_map)
        if cont is False:
            return False

    logger = getattr(trainer, "logger", None)
    if logger is not None and metrics_map:
        try:
            logger.log(
                metrics_map,
                step=int(getattr(trainer, "global_env_step", 0)),
                pbar=pbar,
                prefix="train/",
            )
        except Exception:
            pass

    return True


def _set_global_pbar_postfix(trainer: Any, pbar: Any) -> None:
    """Update global progress postfix with aggregate episode statistics.

    Parameters
    ----------
    trainer : Any
        Trainer-like object exposing episode counters/return helpers.
    pbar : Any
        Progress bar object supporting ``set_postfix``.
    """
    try:
        if int(getattr(trainer, "n_envs", 1)) > 1:
            return
        if pbar is None:
            return
        ep = int(getattr(trainer, "_global_episode_count", 0))
        avg = float(getattr(trainer, "global_avg_return", lambda: 0.0)())
        pbar.set_postfix({"ep": ep, "ret": f"{avg:.4g}"}, refresh=False)
    except Exception:
        pass


def _render_text_dashboard(
    trainer: Any,
    *,
    per_worker: int,
    agent_step_counts: Mapping[str, int],
    agent_last_progress_ts: Mapping[str, float],
    global_step: int,
    window_elapsed_sec: float,
    final: bool,
) -> None:
    """Update Ray training progress-bar postfix from collection-window stats.

    Parameters
    ----------
    trainer : Any
        Trainer-like object providing counters and return helpers.
    per_worker : int
        Target steps per worker for the active collection window.
    agent_step_counts : Mapping[str, int]
        Per-agent progress in current collection window.
    agent_last_progress_ts : Mapping[str, float]
        Per-agent last-progress timestamps used for stall detection.
    global_step : int
        Current global environment step.
    window_elapsed_sec : float
        Elapsed seconds for the current rollout window.
    final : bool
        Whether this render is the final frame for the current window.

    Returns
    -------
    None

    Notes
    -----
    - This function used to render an ASCII dashboard. It now writes rich
      postfix fields to the main tqdm progress bar.
    - Postfix keys include:
      - ``ep``: global episode count,
      - ``ret50``: recent 50-episode average return,
      - ``sps``: global steps-per-second,
      - ``eta``: estimated remaining wall time,
      - ``agent_min%``: minimum per-agent progress in active window,
      - ``stalled``: number of agents considered stalled,
      - ``progress%``: global completion percentage.
    - Rendering is best-effort; any exception is suppressed.
    """
    try:
        n_envs = int(getattr(trainer, "n_envs", 0))
        total_env_steps = max(1, int(getattr(trainer, "total_env_steps", 1)))
        episodes = int(getattr(trainer, "_global_episode_count", 0))
        ret50 = float(getattr(trainer, "recent_avg_return_window", lambda _w=50: 0.0)(50))
        train_pct = 100.0 * float(max(0, global_step)) / float(total_env_steps)
        start_ts = float(getattr(trainer, "_train_start_time", time.time()))
        now_ts = float(time.time())
        elapsed_sec = max(0.0, now_ts - start_ts)
        global_sps = float(global_step) / elapsed_sec if elapsed_sec > 1e-9 else 0.0
        if global_step > 0:
            sec_per_step = elapsed_sec / float(global_step)
            remain_steps = max(0, int(total_env_steps - global_step))
            eta_sec = sec_per_step * float(remain_steps)
        else:
            eta_sec = float("inf")

        denom = max(1, int(per_worker))
        stall_sec = _AGENT_STALL_SEC
        stalled_count = 0
        min_agent_pct = 100.0
        for i in range(n_envs):
            agent_id = f"agent_{i:02d}"
            n = int(agent_step_counts.get(agent_id, 0))
            pct = 100.0 * float(n) / float(denom)
            min_agent_pct = min(min_agent_pct, pct)
            last_prog_ts = float(agent_last_progress_ts.get(agent_id, now_ts))
            stalled = (n < denom) and ((now_ts - last_prog_ts) >= stall_sec)
            if stalled:
                stalled_count += 1

        pbar = getattr(trainer, "_pbar", None)
        if pbar is not None:
            pbar.set_postfix(
                {
                    "ep": episodes,
                    "ret50": f"{ret50:.2f}",
                    "sps": f"{global_sps:.1f}",
                    "eta": _fmt_dur(eta_sec),
                    "agent_min%": f"{min_agent_pct:.1f}",
                    "stalled": stalled_count,
                    "progress%": f"{train_pct:.1f}",
                },
                refresh=bool(final),
            )
    except Exception:
        pass


def _fmt_dur(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration; returns ``"--:--:--"`` for non-finite values.
    """
    if not np.isfinite(float(seconds)):
        return "--:--:--"
    s = int(max(0.0, float(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _maybe_log_sys_ray(trainer: Any, last_sys_log_step: int, pbar: Any) -> int:
    """
    Periodically log system-level counters from the Ray loop.

    Parameters
    ----------
    trainer : Any
        Trainer-like object with logger and counters.
    last_sys_log_step : int
        Last env_step at which sys log was emitted.
    pbar : Any
        Progress bar-like object passed into logger.

    Returns
    -------
    new_last_sys_log_step : int
        Updated last log step (unchanged if no log emitted).

    Notes
    -----
    Payload keys (under prefix "sys/"):
      - env_step
      - update_step
      - episode (only if episode stats are trainer-owned)
    """
    logger = getattr(trainer, "logger", None)
    log_every = int(getattr(trainer, "log_every_steps", 0))
    if logger is None or log_every <= 0:
        return last_sys_log_step

    now = int(getattr(trainer, "global_env_step", 0))
    if (now - int(last_sys_log_step)) < log_every:
        return last_sys_log_step

    payload: Dict[str, Any] = {
        "env_step": int(getattr(trainer, "global_env_step", 0)),
        "update_step": int(getattr(trainer, "global_update_step", 0)),
    }
    try:
        n_envs = int(getattr(trainer, "n_envs", 0))
        per_worker = max(1, int(getattr(trainer, "_ray_per_worker_target", 1)))
        counts = getattr(trainer, "_ray_agent_step_counts_window", {}) or {}
        last_ts_map = getattr(trainer, "_ray_agent_last_progress_ts", {}) or {}
        now_ts = float(time.time())
        stalled_count = 0
        for i in range(n_envs):
            agent_id = f"agent_{i:02d}"
            n = int(counts.get(agent_id, 0))
            last_prog_ts = float(last_ts_map.get(agent_id, now_ts))
            stalled = (n < per_worker) and ((now_ts - last_prog_ts) >= _AGENT_STALL_SEC)
            if stalled:
                stalled_count += 1
        payload["agent_health/stalled_count"] = int(stalled_count)
    except Exception:
        pass
    if not bool(getattr(trainer, "_episode_stats_enabled", False)):
        payload["episode"] = int(getattr(trainer, "episode_idx", 0))

    try:
        logger.log(payload, step=now, pbar=pbar, prefix="sys/")
    except Exception:
        pass

    return now
