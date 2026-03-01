"""Training core for offline Conservative Q-Learning (CQL).

This module implements the optimization logic for continuous-action CQL.
It extends the common actor-critic core utilities with:

- SAC-style actor and entropy-temperature updates.
- Bellman regression with twin critics.
- CQL conservative penalty over out-of-distribution action samples.
- Optional adaptive tuning of the CQL penalty coefficient.
"""


from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from rllib.model_free.common.optimizers.optimizer_builder import build_optimizer
from rllib.model_free.common.optimizers.scheduler_builder import build_scheduler
from rllib.model_free.common.policies.base_core import ActorCriticCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _cql_conservative_loss, _get_per_weights


class CQLCore(ActorCriticCore):
    """Optimization core for continuous-action CQL.

    The core assumes a SAC-compatible head exposing:

    - ``sample_action_and_logp``
    - ``q_values`` and ``q_values_target``
    - ``actor``, ``critic``, and ``critic_target`` modules

    Notes
    -----
    This class does not manage replay storage or environment stepping; those are
    handled by the common ``OffPolicyAlgorithm`` wrapper.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        cql_n_actions: int = 10,
        cql_temp: float = 1.0,
        cql_alpha: float = 1.0,
        cql_target_action_gap: float = 0.0,
        auto_cql_alpha: bool = False,
        cql_alpha_init: float = 1.0,
        cql_alpha_optim_name: str = "adamw",
        cql_alpha_lr: float = 3e-4,
        cql_alpha_weight_decay: float = 0.0,
        cql_alpha_sched_name: str = "none",
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        alpha_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        """Initialize CQL optimization state and auxiliary optimizers.

        Parameters
        ----------
        head : Any
            SAC-compatible policy head owning actor/critic/target modules.
        gamma : float, default=0.99
            Discount factor used in Bellman target computation.
        tau : float, default=0.005
            Polyak averaging coefficient for critic target updates.
        target_update_interval : int, default=1
            Number of update steps between target-network updates.
        auto_alpha : bool, default=True
            Enable automatic entropy-temperature tuning.
        alpha_init : float, default=0.2
            Initial entropy-temperature value.
        target_entropy : float, optional
            Desired policy entropy; if ``None`` uses ``-action_dim``.
        cql_n_actions : int, default=10
            Number of random/policy actions sampled per state for conservative
            penalty estimation.
        cql_temp : float, default=1.0
            Temperature for the CQL log-sum-exp conservative objective.
        cql_alpha : float, default=1.0
            Fixed conservative penalty weight when ``auto_cql_alpha`` is disabled.
        cql_target_action_gap : float, default=0.0
            Target gap value used when adaptively tuning the CQL penalty weight.
        auto_cql_alpha : bool, default=False
            If ``True``, learns the conservative penalty multiplier.
        cql_alpha_init : float, default=1.0
            Initial value for the adaptive conservative multiplier.
        cql_alpha_optim_name, cql_alpha_lr, cql_alpha_weight_decay, cql_alpha_sched_name
            Optimizer/scheduler configuration for adaptive CQL alpha.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold. ``0`` disables clipping.
        use_amp : bool, default=False
            Mixed precision toggle forwarded to the base core.
        actor_optim_name, actor_lr, actor_weight_decay
            Optimizer configuration for actor parameters.
        critic_optim_name, critic_lr, critic_weight_decay
            Optimizer configuration for critic parameters.
        alpha_optim_name, alpha_lr, alpha_weight_decay
            Optimizer configuration for entropy-temperature parameter.
        actor_sched_name, critic_sched_name, alpha_sched_name : str
            Scheduler names for actor/critic/temperature optimizers.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Shared scheduler hyperparameters.

        Returns
        -------
        None
            Initializes internal tensors, optimizers, and schedulers in place.

        Raises
        ------
        ValueError
            If discounting, target update, CQL sampling, or clipping hyperparameters
            are outside valid ranges.
        """
        super().__init__(
            head=head,
            use_amp=bool(use_amp),
            actor_optim_name=str(actor_optim_name),
            actor_lr=float(actor_lr),
            actor_weight_decay=float(actor_weight_decay),
            critic_optim_name=str(critic_optim_name),
            critic_lr=float(critic_lr),
            critic_weight_decay=float(critic_weight_decay),
            actor_sched_name=str(actor_sched_name),
            critic_sched_name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            sched_gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)

        self.cql_n_actions = int(cql_n_actions)
        self.cql_temp = float(cql_temp)
        self.cql_alpha = float(cql_alpha)
        self.cql_target_action_gap = float(cql_target_action_gap)
        self.auto_cql_alpha = bool(auto_cql_alpha)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.cql_n_actions <= 0:
            raise ValueError(f"cql_n_actions must be > 0, got {self.cql_n_actions}")
        if self.cql_temp <= 0.0:
            raise ValueError(f"cql_temp must be > 0, got {self.cql_temp}")
        if self.cql_alpha < 0.0:
            raise ValueError(f"cql_alpha must be >= 0, got {self.cql_alpha}")

        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")
        self.log_alpha = th.tensor(float(math.log(alpha_init)), device=self.device, requires_grad=bool(auto_alpha))
        self.auto_alpha = bool(auto_alpha)

        self.alpha_opt = None
        self.alpha_sched = None
        if self.auto_alpha:
            self.alpha_opt = build_optimizer(
                [self.log_alpha],
                name=str(alpha_optim_name),
                lr=float(alpha_lr),
                weight_decay=float(alpha_weight_decay),
            )
            self.alpha_sched = build_scheduler(
                self.alpha_opt,
                name=str(alpha_sched_name),
                total_steps=int(total_steps),
                warmup_steps=int(warmup_steps),
                min_lr_ratio=float(min_lr_ratio),
                poly_power=float(poly_power),
                step_size=int(step_size),
                gamma=float(sched_gamma),
                milestones=tuple(int(m) for m in milestones),
            )

        self.log_cql_alpha = th.tensor(
            float(math.log(max(cql_alpha_init, 1e-8))),
            device=self.device,
            requires_grad=self.auto_cql_alpha,
        )
        self.cql_alpha_opt = None
        self.cql_alpha_sched = None
        if self.auto_cql_alpha:
            self.cql_alpha_opt = build_optimizer(
                [self.log_cql_alpha],
                name=str(cql_alpha_optim_name),
                lr=float(cql_alpha_lr),
                weight_decay=float(cql_alpha_weight_decay),
            )
            self.cql_alpha_sched = build_scheduler(
                self.cql_alpha_opt,
                name=str(cql_alpha_sched_name),
                total_steps=int(total_steps),
                warmup_steps=int(warmup_steps),
                min_lr_ratio=float(min_lr_ratio),
                poly_power=float(poly_power),
                step_size=int(step_size),
                gamma=float(sched_gamma),
                milestones=tuple(int(m) for m in milestones),
            )

        self.head.freeze_target(self.head.critic_target)

    @property
    def alpha(self) -> th.Tensor:
        """Return current entropy-temperature value.

        Returns
        -------
        torch.Tensor
            Positive scalar tensor ``exp(log_alpha)`` used in SAC actor and target
            value terms.
        """
        return self.log_alpha.exp()

    @property
    def cql_alpha_value(self) -> th.Tensor:
        """Return current conservative-penalty multiplier.

        Returns
        -------
        torch.Tensor
            Scalar tensor controlling CQL penalty magnitude. When adaptive CQL
            alpha is enabled, this is ``exp(log_cql_alpha)`` (clamped at ``0``);
            otherwise it is the fixed configured ``cql_alpha``.
        """
        if self.auto_cql_alpha:
            return self.log_cql_alpha.exp().clamp(min=0.0)
        return th.tensor(self.cql_alpha, device=self.device)

    def _sample_uniform_actions(self, batch_size: int, action_dim: int) -> th.Tensor:
        """Sample uniformly random actions in the valid action range.

        Parameters
        ----------
        batch_size : int
            Number of action vectors to sample.
        action_dim : int
            Action dimension per sampled vector.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, action_dim)`` with values in
            ``[-1, 1]``, matching the squashed-action support used by the head.
        """
        return th.empty((batch_size, action_dim), device=self.device).uniform_(-1.0, 1.0)

    def _compute_cql_penalty(
        self,
        *,
        obs: th.Tensor,
        act: th.Tensor,
        per_w: th.Tensor | None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute CQL conservative regularization for both critics.

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations with shape ``(B, obs_dim)``.
        act : torch.Tensor
            Dataset actions with shape ``(B, action_dim)``.
        per_w : torch.Tensor or None
            Optional per-sample importance weights from prioritized replay.

        Returns
        -------
        tuple of torch.Tensor
            Two scalar tensors ``(penalty, raw)`` where:

            - ``raw`` is the unscaled conservative objective (sum of twin-critic terms).
            - ``penalty`` is ``cql_alpha_value * (raw - cql_target_action_gap)``.

        Notes
        -----
        Out-of-distribution action sets are formed by concatenating:

        - uniformly random actions in ``[-1, 1]``
        - current-policy sampled actions at repeated observations
        """
        bsz, act_dim = int(obs.shape[0]), int(act.shape[1])
        n = self.cql_n_actions

        obs_rep = obs.unsqueeze(1).expand(-1, n, -1).reshape(bsz * n, -1)

        rand_a = self._sample_uniform_actions(batch_size=bsz * n, action_dim=act_dim)
        pi_a, _ = self.head.sample_action_and_logp(obs_rep)

        q1_data, q2_data = self.head.q_values(obs, act)  # (B,1)
        q1_rand, q2_rand = self.head.q_values(obs_rep, rand_a)
        q1_pi, q2_pi = self.head.q_values(obs_rep, pi_a)

        q1_ood = th.cat([q1_rand.view(bsz, n), q1_pi.view(bsz, n)], dim=1)
        q2_ood = th.cat([q2_rand.view(bsz, n), q2_pi.view(bsz, n)], dim=1)

        cql1 = _cql_conservative_loss(q_data=q1_data, q_ood=q1_ood, temperature=self.cql_temp, weights=per_w)
        cql2 = _cql_conservative_loss(q_data=q2_data, q_ood=q2_ood, temperature=self.cql_temp, weights=per_w)

        raw = cql1 + cql2
        alpha_cql = self.cql_alpha_value
        penalty = alpha_cql * (raw - self.cql_target_action_gap)
        return penalty, raw

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Run one full CQL optimization step from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch containing ``observations``, ``actions``, ``rewards``,
            ``next_observations``, and ``dones``. Optional PER fields are consumed
            when available.

        Returns
        -------
        dict[str, Any]
            Metrics dictionary including loss terms, alpha values, learning rates,
            Q/log-prob statistics, and ``per/td_errors`` for priority updates.

        Notes
        -----
        Update order:

        1. Build SAC target and optimize critic with Bellman + CQL penalty.
        2. Optimize actor using entropy-regularized objective.
        3. Optionally update entropy temperature ``alpha``.
        4. Optionally update adaptive CQL multiplier ``cql_alpha``.
        5. Soft-update target critics per configured interval.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        with th.no_grad():
            nxt_a, nxt_logp = self.head.sample_action_and_logp(nxt)
            nxt_logp = _to_column(nxt_logp)
            q1_t, q2_t = self.head.q_values_target(nxt, nxt_a)
            q_min_t = th.min(q1_t, q2_t)
            target_q = rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * nxt_logp)

        q1, q2 = self.head.q_values(obs, act)
        bellman_per = F.mse_loss(q1, target_q, reduction="none") + F.mse_loss(q2, target_q, reduction="none")
        bellman_loss = bellman_per.mean() if w is None else (w * bellman_per).mean()

        cql_penalty, cql_raw = self._compute_cql_penalty(obs=obs, act=act, per_w=w)
        critic_loss = bellman_loss + cql_penalty

        td_abs = (th.min(q1, q2) - target_q).abs().detach().squeeze(1)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
        self.critic_opt.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

        new_a, logp = self.head.sample_action_and_logp(obs)
        logp = _to_column(logp)
        q1_pi, q2_pi = self.head.q_values(obs, new_a)
        q_pi = th.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
        self.actor_opt.step()
        if self.actor_sched is not None:
            self.actor_sched.step()

        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            if self.alpha_sched is not None:
                self.alpha_sched.step()
            alpha_loss_val = float(_to_scalar(alpha_loss))

        cql_alpha_loss_val = 0.0
        if self.cql_alpha_opt is not None:
            cql_alpha_loss = -(self.log_cql_alpha * (cql_raw.detach() - self.cql_target_action_gap)).mean()
            self.cql_alpha_opt.zero_grad(set_to_none=True)
            cql_alpha_loss.backward()
            self.cql_alpha_opt.step()
            if self.cql_alpha_sched is not None:
                self.cql_alpha_sched.step()
            cql_alpha_loss_val = float(_to_scalar(cql_alpha_loss))

        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        out: Dict[str, Any] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/critic_bellman": float(_to_scalar(bellman_loss)),
            "loss/critic_cql": float(_to_scalar(cql_penalty)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "loss/cql_alpha": float(cql_alpha_loss_val),
            "alpha": float(_to_scalar(self.alpha)),
            "cql_alpha": float(_to_scalar(self.cql_alpha_value)),
            "q/q1_mean": float(_to_scalar(q1.mean())),
            "q/q2_mean": float(_to_scalar(q2.mean())),
            "q/pi_min_mean": float(_to_scalar(q_pi.mean())),
            "logp_mean": float(_to_scalar(logp.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])
        if self.cql_alpha_opt is not None:
            out["lr/cql_alpha"] = float(self.cql_alpha_opt.param_groups[0]["lr"])

        return out

    def state_dict(self) -> Dict[str, Any]:
        """Serialize CQL core state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Nested mapping including base core state plus CQL-specific fields
            such as entropy temperature tensors, adaptive-CQL tensors, and
            optimizer/scheduler states when enabled.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(_to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "auto_alpha": bool(self.auto_alpha),
                "target_entropy": float(self.target_entropy),
                "cql_n_actions": int(self.cql_n_actions),
                "cql_temp": float(self.cql_temp),
                "cql_alpha_fixed": float(self.cql_alpha),
                "cql_target_action_gap": float(self.cql_target_action_gap),
                "auto_cql_alpha": bool(self.auto_cql_alpha),
                "log_cql_alpha": float(_to_scalar(self.log_cql_alpha)),
                "cql_alpha_state": (
                    self._save_opt_sched(self.cql_alpha_opt, self.cql_alpha_sched)
                    if self.cql_alpha_opt is not None
                    else None
                ),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore CQL core state from serialized checkpoint data.

        Parameters
        ----------
        state : Mapping[str, Any]
            Mapping produced by :meth:`state_dict`.

        Returns
        -------
        None
            Loads values into in-memory tensors/optimizers in place.

        Notes
        -----
        Missing keys are tolerated to preserve backward compatibility with older
        checkpoints that may not contain all CQL-related fields.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))
        if "auto_alpha" in state:
            self.auto_alpha = bool(state["auto_alpha"])
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)

        if "log_cql_alpha" in state:
            with th.no_grad():
                self.log_cql_alpha.copy_(th.tensor(float(state["log_cql_alpha"]), device=self.device))
        if "auto_cql_alpha" in state:
            self.auto_cql_alpha = bool(state["auto_cql_alpha"])

        cql_alpha_state = state.get("cql_alpha_state", None)
        if self.cql_alpha_opt is not None and isinstance(cql_alpha_state, Mapping):
            self._load_opt_sched(self.cql_alpha_opt, self.cql_alpha_sched, cql_alpha_state)
