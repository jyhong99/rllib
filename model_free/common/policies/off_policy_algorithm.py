"""Off-policy training driver with replay/PER/HER scheduling support.

This module provides :class:`OffPolicyAlgorithm`, a generic environment-facing
driver for replay-based RL methods (e.g., DQN, DDPG, TD3, SAC, TQC, DRQN-style
variants depending on head/core/buffer composition).

Key responsibilities:

- ingest transitions into replay storage,
- perform warmup and update gating,
- manage update-to-data budget accounting,
- sample and forward batches to the learner core,
- integrate optional PER/HER behavior,
- aggregate and normalize scalar logging metrics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import inspect

import numpy as np
import torch as th

from rllib.model_free.common.buffers import HindsightReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
from rllib.model_free.common.utils.common_utils import (
    _infer_shape,
    _require_scalar_like,
    _to_action_np,
    _to_flat_np,
    _to_numpy,
)
from rllib.model_free.common.utils.policy_utils import _infer_n_actions_from_env
from rllib.model_free.common.policies.base_policy import BasePolicyAlgorithm


class OffPolicyAlgorithm(BasePolicyAlgorithm):
    """
    Generic off-policy algorithm driver (SAC/TD3/DDPG/DQN-family style).

    This driver composes three duck-typed components:

    - **head**: action selection (exploration/noise typically lives here)
    - **core**: gradient updates given replay batches (`core.update_from_batch(batch)`)
    - **buffer**: replay storage (uniform ReplayBuffer or PrioritizedReplayBuffer)

    Scheduling semantics
    -------------------
    Warmup (`update_after`)
      - For `env_steps < update_after`: actions are sampled randomly and updates
        are disabled.

    Update gating (`update_every`)
      - Updates are only allowed when `env_steps % update_every == 0` (if `update_every > 1`).

    UTD update budget (`utd`)
      - After warmup, each env step accrues an "update budget":
          update_budget += utd * n_new_steps
      - Updates consume 1.0 budget per "update unit".
      - IMPORTANT: during warmup (env_steps < update_after), we do NOT accrue
        budget. This avoids a large backlog of updates firing immediately after
        warmup completes.

    Gradient steps per update unit (`gradient_steps`)
      - Each update unit performs `gradient_steps` SGD steps. Metrics are averaged.

    Notes
    -----
    - This class does not define the loss; it delegates to `core.update_from_batch`.
    - For DQN-style epsilon-greedy exploration, if `head.act` accepts `epsilon`,
      the driver passes epsilon; otherwise it calls `head.act` without epsilon.
    - Supports optional behavior-policy statistics storage (logp/probs) inside the buffer.
    - Supports n-step replay if `n_step > 1` and the buffer implementation supports it.
    """

    is_off_policy: bool = True

    # Metric keys that (optionally) may be returned by the core for PER.
    _PRIORITIES_KEY = "per/priorities"  # explicit new priorities
    _TD_ERRORS_KEY = "per/td_errors"    # td-errors; will be abs()'d into priorities

    @staticmethod
    def _linear_anneal(step: int, total_steps: int, start: float, end: float) -> float:
        """
        Compute a linear annealed value between two endpoints.

        Parameters
        ----------
        step : int
            Current step index.
        total_steps : int
            Number of steps over which to anneal. If ``<= 0``, ``start`` is
            returned directly.
        start : float
            Initial value at step 0.
        end : float
            Final value at and after ``total_steps``.

        Returns
        -------
        value : float
            Interpolated value at ``step`` after clamping ``step`` into
            ``[0, total_steps]``.
        """
        if total_steps <= 0:
            return float(start)
        t = min(max(int(step), 0), int(total_steps))
        frac = float(t) / float(total_steps)
        return float(start + frac * (end - start))

    @staticmethod
    def _td_error_stats_from_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
        """
        Extract scalar TD-error summary stats from update metrics.

        Parameters
        ----------
        metrics : Mapping[str, Any]
            Raw metrics mapping that may include ``per/td_errors`` as an array.

        Returns
        -------
        Dict[str, float]
            Optional keys:
            - ``td_error_mean``
            - ``td_error_p95``
        """
        td_raw = metrics.get(OffPolicyAlgorithm._TD_ERRORS_KEY, None)
        if td_raw is None:
            return {}

        try:
            td_np = np.asarray(_to_flat_np(td_raw), dtype=np.float32).reshape(-1)
        except Exception:
            return {}

        if td_np.size <= 0:
            return {}

        td_abs = np.abs(td_np)
        td_abs = td_abs[np.isfinite(td_abs)]
        if td_abs.size <= 0:
            return {}

        return {
            "td_error_mean": float(np.mean(td_abs)),
            "td_error_p95": float(np.percentile(td_abs, 95.0)),
        }

    def _replay_fill_ratio(self) -> float:
        """
        Return replay occupancy ratio.

        Returns
        -------
        ratio : float
            ``buffer.size / buffer.capacity`` clamped to ``[0, 1]``. Returns
            ``0.0`` when replay is uninitialized or capacity is invalid.
        """
        if self.buffer is None:
            return 0.0
        cap = float(getattr(self.buffer, "capacity", 0))
        if cap <= 0.0:
            return 0.0
        size = float(getattr(self.buffer, "size", 0))
        return float(max(0.0, min(1.0, size / cap)))

    def __init__(
        self,
        *,
        head: Any,
        core: Any,
        # replay + schedule
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        update_after: int = 1_000,
        update_every: int = 1,
        utd: float = 1.0,
        gradient_steps: int = 1,
        max_updates_per_call: int = 1_000,
        # storage
        device: Optional[Union[str, th.device]] = None,
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
        # PER
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        per_beta_final: float = 1.0,
        per_beta_anneal_steps: int = 200_000,
        # behavior stats storage (useful for off-policy corrections / logging)
        store_behavior_logp: bool = False,
        store_behavior_probs: bool = False,
        # n-step
        n_step: int = 1,
        # HER
        use_her: bool = False,
        her_goal_shape: Optional[Tuple[int, ...]] = None,
        her_reward_fn: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        her_done_fn: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        her_ratio: float = 0.8,
        # sequence replay (for recurrent policies)
        replay_sequence_length: int = 1,
        replay_sequence_strict_done: bool = True,
        # head-side noise reset policy
        reset_noise_on_done: bool = False,
        # epsilon-greedy exploration (for DQN/Q-learning heads)
        exploration_eps: float = 0.1,
        exploration_eps_final: float = 0.05,
        exploration_eps_anneal_steps: int = 200_000,
        exploration_eval_eps: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Action-selection head (duck-typed). Expected to expose:
            - act(obs, deterministic=..., epsilon=...) depending on head type
            - reset_exploration_noise() optional
        core : Any
            Update engine (duck-typed). Must expose:
            - update_from_batch(batch) -> Mapping[str, Any] (metrics)
            Optional:
            - gamma attribute (used for n-step replay config fallback)
        buffer_size : int, default=1_000_000
            Replay buffer capacity (number of transitions).
        batch_size : int, default=256
            Minibatch size sampled from replay for one gradient step.
        update_after : int, default=1_000
            Warmup horizon in env steps. No learning updates before this.
        update_every : int, default=1
            Update gating period. If > 1, updates only when `env_steps % update_every == 0`.
        utd : float, default=1.0
            Updates-to-data ratio. Controls how much update budget accrues per env step.
        gradient_steps : int, default=1
            Number of SGD steps per update unit.
        max_updates_per_call : int, default=1_000
            Hard cap on the number of update units performed in a single `update()` call
            (prevents extremely long update bursts).
        device : Optional[Union[str, torch.device]], optional
            Storage/sampling device for the replay buffer (depends on buffer implementation).
        dtype_obs : Any, default=np.float32
            Storage dtype for observations.
        dtype_act : Any, default=np.float32
            Storage dtype for actions.
        use_per : bool, default=False
            If True, use Prioritized Experience Replay.
        per_alpha : float, default=0.6
            PER prioritization exponent.
        per_beta : float, default=0.4
            Initial importance-sampling exponent. Must be in [0, 1].
        per_eps : float, default=1e-6
            Small constant to avoid zero priorities.
        per_beta_final : float, default=1.0
            Final beta after annealing. Must be in [0, 1].
        per_beta_anneal_steps : int, default=200_000
            Number of env steps over which beta is annealed to per_beta_final.
        store_behavior_logp : bool, default=False
            If True, store behavior log-probabilities in replay (requires transition key).
        store_behavior_probs : bool, default=False
            If True, store behavior action probabilities (discrete policies).
        n_step : int, default=1
            N-step return length. If > 1, passes n-step config into replay buffer.
        reset_noise_on_done : bool, default=False
            If True, calls head.reset_exploration_noise() when episode ends.
        exploration_eps : float, default=0.1
            Initial epsilon for epsilon-greedy (DQN-style).
        exploration_eps_final : float, default=0.05
            Final epsilon after annealing.
        exploration_eps_anneal_steps : int, default=200_000
            Number of env steps to linearly anneal epsilon.
        exploration_eval_eps : float, default=0.0
            Epsilon used when deterministic=True.

        Raises
        ------
        ValueError
            If hyperparameters are invalid.
        """
        super().__init__(head=head, core=core, device=device)

        # schedule / sizing
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.utd = float(utd)
        self.gradient_steps = int(gradient_steps)
        self.max_updates_per_call = int(max_updates_per_call)

        # storage dtypes
        self.dtype_obs = dtype_obs
        self.dtype_act = dtype_act

        # PER hyperparams
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        self.per_beta_final = float(per_beta_final)
        self.per_beta_anneal_steps = int(per_beta_anneal_steps)

        # optional behavior storage
        self.store_behavior_logp = bool(store_behavior_logp)
        self.store_behavior_probs = bool(store_behavior_probs)
        self.n_actions: Optional[int] = None  # inferred in setup() if needed

        # n-step
        self.n_step = int(n_step)
        self.n_step_gamma = float(getattr(self.core, "gamma", 0.99))
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got: {self.n_step}")

        self.use_her = bool(use_her)
        self.her_goal_shape = tuple(int(x) for x in her_goal_shape) if her_goal_shape is not None else None
        self.her_reward_fn = her_reward_fn
        self.her_done_fn = her_done_fn
        self.her_ratio = float(her_ratio)

        self.replay_sequence_length = int(replay_sequence_length)
        self.replay_sequence_strict_done = bool(replay_sequence_strict_done)
        if self.replay_sequence_length < 1:
            raise ValueError(
                f"replay_sequence_length must be >= 1, got: {self.replay_sequence_length}"
            )

        # head-side noise reset policy
        self.reset_noise_on_done = bool(reset_noise_on_done)

        # epsilon exploration config (for DQN/Q-learning heads)
        self.exploration_eps = float(exploration_eps)
        self.exploration_eps_final = float(exploration_eps_final)
        self.exploration_eps_anneal_steps = int(exploration_eps_anneal_steps)
        self.exploration_eval_eps = float(exploration_eval_eps)

        if self.exploration_eps < 0.0:
            raise ValueError(f"exploration_eps must be >= 0, got {self.exploration_eps}")
        if self.exploration_eps_final < 0.0:
            raise ValueError(f"exploration_eps_final must be >= 0, got {self.exploration_eps_final}")
        if self.exploration_eps_anneal_steps < 0:
            raise ValueError(f"exploration_eps_anneal_steps must be >= 0, got {self.exploration_eps_anneal_steps}")

        # Cache whether head.act supports epsilon to avoid repeated introspection
        self._head_act_accepts_epsilon = False
        try:
            sig = inspect.signature(self.head.act)
            self._head_act_accepts_epsilon = ("epsilon" in sig.parameters)
        except Exception:
            self._head_act_accepts_epsilon = False

        # runtime state (initialized in setup)
        self.buffer: Optional[Union[ReplayBuffer, PrioritizedReplayBuffer, HindsightReplayBuffer]] = None
        self._action_space: Optional[Any] = None
        self._action_shape: Optional[tuple[int, ...]] = None

        # Warmup sampling mode: if env expects actions in policy space (e.g., [-1, 1])
        self._warmup_policy_action_space: bool = False

        # fractional update budget (accrues only after warmup)
        self._update_budget: float = 0.0

        self._validate_hparams()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_hparams(self) -> None:
        """
        Validate constructor hyperparameters.

        Raises
        ------
        ValueError
            If any schedule, replay, PER, HER, or exploration parameter is
            outside its supported range, or unsupported feature combinations are
            requested.
        """
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got: {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {self.batch_size}")
        if self.update_after < 0:
            raise ValueError(f"update_after must be >= 0, got: {self.update_after}")
        if self.update_every <= 0:
            raise ValueError(f"update_every must be > 0, got: {self.update_every}")
        if self.gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be > 0, got: {self.gradient_steps}")
        if self.max_updates_per_call <= 0:
            raise ValueError(f"max_updates_per_call must be > 0, got: {self.max_updates_per_call}")
        if self.utd < 0.0:
            raise ValueError(f"utd must be >= 0, got: {self.utd}")

        if self.use_per:
            if self.use_her:
                raise ValueError("use_her=True is not supported with use_per=True.")
            if self.per_alpha < 0.0:
                raise ValueError(f"per_alpha must be >= 0, got: {self.per_alpha}")
            if not (0.0 <= self.per_beta <= 1.0):
                raise ValueError(f"per_beta must be in [0, 1], got: {self.per_beta}")
            if not (0.0 <= self.per_beta_final <= 1.0):
                raise ValueError(f"per_beta_final must be in [0, 1], got: {self.per_beta_final}")
            if self.per_eps <= 0.0:
                raise ValueError(f"per_eps must be > 0, got: {self.per_eps}")
            if self.per_beta_anneal_steps < 0:
                raise ValueError(f"per_beta_anneal_steps must be >= 0, got: {self.per_beta_anneal_steps}")

        if self.use_her:
            if self.n_step != 1:
                raise ValueError("use_her=True currently requires n_step=1.")
            if self.replay_sequence_length != 1:
                raise ValueError("use_her=True currently requires replay_sequence_length=1.")
            if not (0.0 <= self.her_ratio <= 1.0):
                raise ValueError(f"her_ratio must be in [0,1], got: {self.her_ratio}")
            if self.her_reward_fn is not None and not callable(self.her_reward_fn):
                raise ValueError("her_reward_fn must be callable when provided.")
            if self.her_done_fn is not None and not callable(self.her_done_fn):
                raise ValueError("her_done_fn must be callable when provided.")
            if self.her_goal_shape is not None and any(int(x) <= 0 for x in self.her_goal_shape):
                raise ValueError(f"her_goal_shape must contain positive dims, got: {self.her_goal_shape}")

    # =============================================================================
    # Epsilon schedule (DQN-style)
    # =============================================================================
    def _current_exploration_eps(self) -> float:
        """
        Current epsilon for epsilon-greedy exploration.

        Returns
        -------
        eps : float
            Linearly annealed epsilon from `exploration_eps` to `exploration_eps_final`
            over `exploration_eps_anneal_steps` env steps. If anneal_steps <= 0,
            returns `exploration_eps`.
        """
        eps = self._linear_anneal(
            step=int(self._env_steps),
            total_steps=int(self.exploration_eps_anneal_steps),
            start=float(self.exploration_eps),
            end=float(self.exploration_eps_final),
        )
        return float(max(eps, 0.0))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, env: Any) -> None:
        """
        Initialize replay buffer and cache env action/observation shapes.

        Parameters
        ----------
        env : Any
            Environment object with `observation_space` and `action_space` fields.
            For PER/discrete behavior probs, `env` must also allow inferring n_actions.

        Notes
        -----
        - Resets:
          - `env_steps` counter
          - update budget
          - optional head-side exploration noise
        - If `n_step > 1`, passes `{n_step, gamma}` into the buffer constructor.
        """
        obs_shape = _infer_shape(env.observation_space, name="observation_space")
        action_shape = _infer_shape(env.action_space, name="action_space")

        self._action_space = env.action_space
        self._action_shape = tuple(int(x) for x in action_shape)

        # If env internally rescales actions into policy space, warmup random actions should
        # match policy range (typically [-1,1]) to avoid OOD actions early on.
        self._warmup_policy_action_space = bool(getattr(env, "action_rescale", False))

        if self.store_behavior_probs:
            self.n_actions = int(_infer_n_actions_from_env(env))

        # Optional n-step configuration (buffer must support it).
        nstep_kwargs: Dict[str, Any] = {}
        if self.n_step > 1:
            nstep_kwargs = {"n_step": int(self.n_step), "gamma": float(self.n_step_gamma)}

        if self.use_her:
            goal_shape = self.her_goal_shape
            if goal_shape is None:
                env_goal_shape = getattr(env, "goal_shape", None)
                if env_goal_shape is not None:
                    goal_shape = tuple(int(x) for x in env_goal_shape)
            if goal_shape is None:
                raise ValueError("use_her=True requires her_goal_shape or env.goal_shape.")

            reward_fn = self.her_reward_fn
            if reward_fn is None:
                reward_fn = getattr(env, "compute_reward", None)
            if reward_fn is None or not callable(reward_fn):
                raise ValueError("use_her=True requires her_reward_fn or env.compute_reward callable.")

            self.buffer = HindsightReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=obs_shape,
                action_shape=action_shape,
                goal_shape=goal_shape,
                reward_fn=reward_fn,
                done_fn=self.her_done_fn,
                her_ratio=self.her_ratio,
                device=self.device,
                dtype_obs=self.dtype_obs,
                dtype_act=self.dtype_act,
            )
        elif self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=obs_shape,
                action_shape=action_shape,
                alpha=self.per_alpha,
                beta=self.per_beta,
                eps=self.per_eps,
                device=self.device,
                dtype_obs=self.dtype_obs,
                dtype_act=self.dtype_act,
                store_behavior_logp=self.store_behavior_logp,
                store_behavior_probs=self.store_behavior_probs,
                n_actions=self.n_actions,
                **nstep_kwargs,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=obs_shape,
                action_shape=action_shape,
                device=self.device,
                dtype_obs=self.dtype_obs,
                dtype_act=self.dtype_act,
                store_behavior_logp=self.store_behavior_logp,
                store_behavior_probs=self.store_behavior_probs,
                n_actions=self.n_actions,
                **nstep_kwargs,
            )

        # Optional head noise reset (common in DDPG/TD3-style exploration).
        reset_fn = getattr(self.head, "reset_exploration_noise", None)
        if callable(reset_fn):
            reset_fn()

        self._env_steps = 0
        self._update_budget = 0.0

    # =============================================================================
    # Acting
    # =============================================================================
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """
        Select an action.

        Parameters
        ----------
        obs : Any
            Observation(s).
        deterministic : bool, default=False
            If True, disables exploration (head-dependent). For epsilon-greedy heads,
            uses `exploration_eval_eps` as epsilon.

        Returns
        -------
        action : Any
            Action in env-native format.

        Notes
        -----
        - Warmup (`env_steps < update_after`): returns random action.
        - After warmup:
          - if `head.act` supports epsilon: calls `head.act(obs, epsilon=..., deterministic=...)`
          - else: calls `head.act(obs, deterministic=...)`
        """
        if self.update_after > 0 and self._env_steps < self.update_after:
            return self._sample_random_action()

        eps = float(self.exploration_eval_eps) if deterministic else float(self._current_exploration_eps())

        if self._head_act_accepts_epsilon:
            return self.head.act(obs, epsilon=eps, deterministic=deterministic)

        return self.head.act(obs, deterministic=deterministic)

    # =============================================================================
    # Data ingestion
    # =============================================================================
    def on_env_step(self, transition: Dict[str, Any]) -> None:
        """
        Consume one transition and push it into replay.

        Parameters
        ----------
        transition : Dict[str, Any]
            Must contain at least:
            - "obs"
            - "action"
            - "reward"
            - "next_obs"
            - "done"
            Optional (depending on config):
            - "behavior_logp"   (if store_behavior_logp=True)
            - "behavior_probs"  (if store_behavior_probs=True)
            - "priority"        (optional initial PER priority)

        Raises
        ------
        RuntimeError
            If setup(env) was not called (buffer is None).
        ValueError
            If required transition keys are missing or have invalid shapes.

        Notes
        -----
        When ``store_behavior_probs=True``, behavior probabilities are validated
        (finite, non-negative, positive sum) and normalized to sum to 1 before
        being stored.
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        self._env_steps += 1

        obs_np = np.asarray(_to_flat_np(transition["obs"]), dtype=self.dtype_obs)
        next_obs_np = np.asarray(_to_flat_np(transition["next_obs"]), dtype=self.dtype_obs)

        reward = _require_scalar_like(transition["reward"], name="transition['reward']")
        done = bool(_require_scalar_like(transition["done"], name="transition['done']"))

        action_np = _to_action_np(transition["action"], action_shape=self._action_shape)
        action_np = np.asarray(action_np, dtype=self.dtype_act)

        beh_logp: Optional[float] = None
        beh_probs: Optional[np.ndarray] = None

        if self.store_behavior_logp:
            if "behavior_logp" not in transition:
                raise ValueError("store_behavior_logp=True requires transition['behavior_logp'].")
            beh_logp = _require_scalar_like(transition["behavior_logp"], name="transition['behavior_logp']")

        if self.store_behavior_probs:
            if "behavior_probs" not in transition:
                raise ValueError("store_behavior_probs=True requires transition['behavior_probs'].")
            if self.n_actions is None:
                raise RuntimeError("n_actions is not set. Did you call setup(env)?")

            bp = np.asarray(_to_numpy(transition["behavior_probs"]), dtype=np.float32).reshape(-1)
            if bp.shape[0] != int(self.n_actions):
                raise ValueError(f"behavior_probs must have shape (A,), got {bp.shape} (A={self.n_actions})")
            if not np.isfinite(bp).all():
                raise ValueError("behavior_probs must be finite.")
            if (bp < 0.0).any():
                raise ValueError("behavior_probs must be non-negative.")
            s = float(bp.sum())
            if s <= 0.0:
                raise ValueError("behavior_probs must sum to a positive value.")
            bp = bp / s
            beh_probs = bp

        priority: Optional[float] = None
        if self.use_per and ("priority" in transition):
            priority = _require_scalar_like(transition["priority"], name="transition['priority']")

        # Push into replay.
        if isinstance(self.buffer, HindsightReplayBuffer):
            if "desired_goal" not in transition:
                raise ValueError("use_her=True requires transition['desired_goal'].")
            if "achieved_goal" not in transition:
                raise ValueError("use_her=True requires transition['achieved_goal'].")
            if "next_achieved_goal" not in transition:
                raise ValueError("use_her=True requires transition['next_achieved_goal'].")

            goal_shape = tuple(int(x) for x in self.buffer.goal_shape)
            desired_goal = np.asarray(_to_numpy(transition["desired_goal"]), dtype=np.float32).reshape(goal_shape)
            achieved_goal = np.asarray(_to_numpy(transition["achieved_goal"]), dtype=np.float32).reshape(goal_shape)
            next_achieved_goal = np.asarray(_to_numpy(transition["next_achieved_goal"]), dtype=np.float32).reshape(goal_shape)

            self.buffer.add(
                obs=obs_np,
                action=action_np,
                reward=reward,
                next_obs=next_obs_np,
                done=done,
                desired_goal=desired_goal,
                achieved_goal=achieved_goal,
                next_achieved_goal=next_achieved_goal,
            )
        elif isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.add(
                obs=obs_np,
                action=action_np,
                reward=reward,
                next_obs=next_obs_np,
                done=done,
                priority=priority,
                behavior_logp=beh_logp,
                behavior_probs=beh_probs,
            )
        else:
            self.buffer.add(
                obs=obs_np,
                action=action_np,
                reward=reward,
                next_obs=next_obs_np,
                done=done,
                behavior_logp=beh_logp,
                behavior_probs=beh_probs,
            )

        # IMPORTANT: do not accrue update budget during warmup (avoids post-warmup backlog).
        if self._env_steps >= self.update_after:
            self._accumulate_update_budget(n_new_steps=1)

        # Optional head-side noise reset at episode boundary.
        if done and self.reset_noise_on_done:
            reset_fn = getattr(self.head, "reset_exploration_noise", None)
            if callable(reset_fn):
                reset_fn()

    def _accumulate_update_budget(self, n_new_steps: int) -> None:
        """
        Accumulate fractional update budget.

        Parameters
        ----------
        n_new_steps : int
            Number of newly collected env steps to convert into update budget.
        """
        if self.utd <= 0.0:
            return
        self._update_budget += self.utd * float(int(n_new_steps))

    # =============================================================================
    # Update scheduling
    # =============================================================================
    def ready_to_update(self) -> bool:
        """
        Check whether an update is allowed at the current env step.

        Returns
        -------
        ready : bool
            True if:
            - buffer is initialized
            - env_steps >= update_after
            - replay has at least batch_size transitions
            - update_budget >= 1.0
            - update_every gating condition passes
        """
        if self.buffer is None:
            return False
        if self._env_steps < self.update_after:
            return False
        if self.replay_sequence_length <= 1:
            if self.buffer.size < self.batch_size:
                return False
        else:
            n_valid_fn = getattr(self.buffer, "num_valid_sequence_starts", None)
            if callable(n_valid_fn):
                if int(n_valid_fn(seq_len=self.replay_sequence_length, strict_done=self.replay_sequence_strict_done)) < self.batch_size:
                    return False
            else:
                if self.buffer.size < self.batch_size:
                    return False
        if self._update_budget < 1.0:
            return False
        if self.update_every > 1 and (self._env_steps % self.update_every) != 0:
            return False
        return True

    # =============================================================================
    # PER helpers
    # =============================================================================
    def _current_per_beta(self) -> float:
        """
        Current PER importance-sampling beta (annealed).

        Returns
        -------
        beta : float
            Linearly annealed beta from per_beta to per_beta_final over
            per_beta_anneal_steps env steps. If anneal_steps <= 0, returns per_beta.
        """
        return self._linear_anneal(
            step=int(self._env_steps),
            total_steps=int(self.per_beta_anneal_steps),
            start=float(self.per_beta),
            end=float(self.per_beta_final),
        )

    @staticmethod
    def _get_batch_indices(batch: Any) -> Optional[np.ndarray]:
        """
        Extract PER indices from a sampled batch.

        Parameters
        ----------
        batch : Any
            Batch object that may expose `.indices`.

        Returns
        -------
        indices : Optional[np.ndarray]
            1D int64 array of indices, or None if not present.
        """
        idx = getattr(batch, "indices", None)
        if idx is None:
            return None
        if th.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        return np.asarray(idx, dtype=np.int64).reshape(-1)

    def _maybe_update_per_priorities(self, *, batch: Any, metrics: Dict[str, Any]) -> None:
        """
        Update PER priorities using information returned by the core.

        The core may return either:
        - `per/priorities`: new priorities directly, shape (B,)
        - `per/td_errors` : td-errors, shape (B,), converted to abs(td_error)

        Parameters
        ----------
        batch : Any
            Sampled batch (must expose indices for PER).
        metrics : Dict[str, Any]
            Metrics dict returned by core. This method will `pop()` the PER-related
            keys from the dict to avoid logging them as generic scalars.
        """
        if not isinstance(self.buffer, PrioritizedReplayBuffer):
            return

        indices = self._get_batch_indices(batch)
        if indices is None or indices.size == 0:
            return

        pr = metrics.pop(self._PRIORITIES_KEY, None)
        td = metrics.pop(self._TD_ERRORS_KEY, None)
        if pr is None and td is None:
            return

        arr = pr if pr is not None else td
        pr_np = np.asarray(_to_flat_np(arr), dtype=np.float32).reshape(-1)

        # If only td-errors are provided, use absolute value as priority proxy.
        if pr is None and td is not None:
            pr_np = np.abs(pr_np)

        if pr_np.shape[0] != indices.shape[0]:
            return

        self.buffer.update_priorities(indices, pr_np)

    # =============================================================================
    # Update primitives
    # =============================================================================
    def update_once(self) -> Dict[str, float]:
        """
        Perform a single *gradient step* update from replay.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar metrics (filtered). For PER, also reports `per/beta` and `per/enabled`.
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            beta = self._current_per_beta()
            batch = self.buffer.sample(
                self.batch_size,
                beta=beta,
                seq_len=self.replay_sequence_length,
                strict_done=self.replay_sequence_strict_done,
            )

            metrics_any = self.core.update_from_batch(batch)
            metrics = dict(metrics_any) if isinstance(metrics_any, Mapping) else {}
            td_stats = self._td_error_stats_from_metrics(metrics)

            self._maybe_update_per_priorities(batch=batch, metrics=metrics)

            out = self._filter_scalar_metrics(metrics)
            out.update(td_stats)
            out["per/beta"] = float(beta)
            out["per/enabled"] = 1.0
            return out

        batch = self.buffer.sample(
            self.batch_size,
            seq_len=self.replay_sequence_length,
            strict_done=self.replay_sequence_strict_done,
        )
        metrics_any = self.core.update_from_batch(batch)
        metrics = dict(metrics_any) if isinstance(metrics_any, Mapping) else {}
        td_stats = self._td_error_stats_from_metrics(metrics)
        out = self._filter_scalar_metrics(metrics)
        out.update(td_stats)
        if isinstance(self.buffer, HindsightReplayBuffer):
            out["her/enabled"] = 1.0
        return out

    def update(self) -> Dict[str, float]:
        """
        Perform scheduled off-policy updates based on accumulated update budget.

        Returns
        -------
        metrics : Dict[str, float]
            Aggregated scalar metrics averaged over:
            - `n_updates` update units
            - `gradient_steps` inner SGD steps per update unit

        Notes
        -----
        - Consumes integer part of `self._update_budget`.
        - Caps updates by `max_updates_per_call`.
        - Logs:
          - buffer size, env steps, updates ran, remaining budget
          - exploration epsilon
          - sys/num_updates = n_updates * gradient_steps
        """
        if self.buffer is None:
            raise RuntimeError("ReplayBuffer not initialized. Call setup(env) first.")

        owed = int(self._update_budget)
        if owed <= 0:
            out: Dict[str, float] = {
                "offpolicy/buffer_size": float(self.buffer.size),
                "replay/fill_ratio": float(self._replay_fill_ratio()),
                "offpolicy/env_steps": float(self._env_steps),
                "offpolicy/update_budget": float(self._update_budget),
            }
            if self.use_per:
                out["per/enabled"] = 1.0
            if isinstance(self.buffer, HindsightReplayBuffer):
                out["her/enabled"] = 1.0
            return out

        n_updates = min(owed, self.max_updates_per_call)
        gs = int(self.gradient_steps)

        agg: Dict[str, float] = {}

        # Outer loop: update units
        for _ in range(n_updates):
            inner: Dict[str, float] = {}

            # Inner loop: gradient steps per unit
            for _g in range(gs):
                m = self.update_once()
                for k, v in m.items():
                    inner[k] = inner.get(k, 0.0) + float(v)

            # Average over gradient steps
            inv_gs = 1.0 / float(gs)
            for k in list(inner.keys()):
                inner[k] *= inv_gs

            # Accumulate into global aggregator
            for k, v in inner.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        # Average over update units
        inv_n = 1.0 / float(n_updates)
        for k in list(agg.keys()):
            agg[k] *= inv_n

        # Consume budget (one per update unit)
        self._update_budget -= float(n_updates)

        # Add driver-level metrics
        agg["offpolicy/buffer_size"] = float(self.buffer.size)
        agg["replay/fill_ratio"] = float(self._replay_fill_ratio())
        agg["offpolicy/env_steps"] = float(self._env_steps)
        agg["offpolicy/updates_ran"] = float(n_updates)
        agg["offpolicy/update_budget"] = float(self._update_budget)

        if self.use_per:
            agg["per/enabled"] = 1.0
        if isinstance(self.buffer, HindsightReplayBuffer):
            agg["her/enabled"] = 1.0

        agg["sys/num_updates"] = float(n_updates * gs)
        agg["offpolicy/grad_steps"] = float(gs)
        agg["offpolicy/update_units"] = float(n_updates)
        agg["exploration/epsilon"] = float(self._current_exploration_eps())

        return agg

    # =============================================================================
    # Random action sampling (warmup = env_steps < update_after)
    # =============================================================================
    def _sample_random_action(self) -> Any:
        """
        Sample a random action from the environment action space.

        Returns
        -------
        action : Any
            Random action in env-native format.

        Notes
        -----
        Handles common Gym/Gymnasium spaces:
        - Box-like: uses uniform sampling in [low, high] (or [-1,1] if warmup policy space).
        - Discrete-like: uses randint in [0, n).
        - Generic spaces with `.sample()`.

        The `_warmup_policy_action_space` flag exists for environments that internally
        rescale actions into policy space (e.g., actor outputs in [-1,1])—in that case,
        warmup should sample from [-1,1] to stay in-distribution for the policy/head.
        """
        if self._action_space is None or self._action_shape is None:
            raise RuntimeError("action_space/action_shape not cached. Did you call setup(env)?")

        space = self._action_space

        # Preferred: delegate to space.sample() if it exists.
        if callable(getattr(space, "sample", None)):
            a = space.sample()

            # If env expects policy-space actions during warmup, override Box sampling.
            if not self._warmup_policy_action_space:
                return a

            if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
                shape = tuple(int(x) for x in space.shape)
                return np.random.uniform(-1.0, 1.0, size=shape).astype(self.dtype_act)

            return a

        # Discrete-like space: has attribute `.n`.
        if hasattr(space, "n"):
            return int(np.random.randint(0, int(space.n)))

        # Box-like fallback: has `.low`, `.high`, `.shape`.
        if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            shape = tuple(int(x) for x in space.shape)

            low = np.broadcast_to(low, shape) if low.shape != shape else low
            high = np.broadcast_to(high, shape) if high.shape != shape else high

            if self._warmup_policy_action_space:
                return np.random.uniform(-1.0, 1.0, size=shape).astype(self.dtype_act)

            return np.random.uniform(low=low, high=high, size=shape).astype(self.dtype_act)

        raise ValueError(f"Unsupported action_space for random sampling: {space}")
