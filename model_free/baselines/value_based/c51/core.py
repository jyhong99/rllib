"""C51 Core.

This module implements the update core for C51 (categorical DQN) in discrete
action spaces.

It includes:

- Categorical Bellman target projection onto fixed support.
- Optional Double-DQN action selection for next-state target distribution.
- PER weighting and TD-error export for priority updates.
- Optimizer/scheduler integration and target-network updates.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th

from rllib.model_free.common.policies.base_core import QLearningCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _distribution_projection, _get_per_weights


class C51Core(QLearningCore):
    """C51 update core for discrete actions.

    Notes
    -----
    The core expects a head with:

    - ``head.q.dist(obs)`` producing distributions of shape ``(B, A, K)``.
    - ``head.q_target.dist(obs)`` with matching atom dimension.
    - ``head.support`` as a 1-D support tensor of shape ``(K,)``.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        double_dqn: bool = True,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        per_eps: float = 1e-6,
        log_eps: float = 1e-6,
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        """Initialize a C51 update core.

        Parameters
        ----------
        head : Any
            Q-learning head exposing online/target distributional Q-networks.
        gamma : float, default=0.99
            Discount factor in ``[0, 1)``.
        target_update_interval : int, default=1000
            Hard update interval when ``tau=0``.
        tau : float, default=0.0
            Soft update coefficient in ``[0, 1]``.
        double_dqn : bool, default=True
            Use online argmax + target gather for next-action selection.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold. ``0`` disables clipping.
        use_amp : bool, default=False
            Enable AMP for backward/optimizer step.
        per_eps : float, default=1e-6
            Minimum TD-error floor exported for PER priorities.
        log_eps : float, default=1e-6
            Minimum clamp value used before ``log`` on predicted categorical
            probabilities.
        optim_name : str, default="adamw"
            Optimizer name for online Q-network updates.
        lr : float, default=3e-4
            Optimizer learning rate.
        weight_decay : float, default=0.0
            Optimizer weight decay.
        sched_name : str, default="none"
            Scheduler name (optional).
        total_steps : int, default=0
            Total training steps for compatible schedulers.
        warmup_steps : int, default=0
            Warmup steps for compatible schedulers.
        min_lr_ratio : float, default=0.0
            Minimum LR ratio for compatible schedulers.
        poly_power : float, default=1.0
            Polynomial power for compatible schedulers.
        step_size : int, default=1000
            Step size for step schedulers.
        sched_gamma : float, default=0.99
            Decay factor for step/exponential schedulers.
        milestones : Sequence[int], default=()
            Milestones for multi-step schedulers.
        """
        super().__init__(
            head=head,
            use_amp=use_amp,
            optim_name=optim_name,
            lr=lr,
            weight_decay=weight_decay,
            sched_name=sched_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            poly_power=poly_power,
            step_size=step_size,
            sched_gamma=sched_gamma,
            milestones=milestones,
        )

        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)
        self.log_eps = float(log_eps)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")
        if self.log_eps <= 0.0:
            raise ValueError(f"log_eps must be > 0, got {self.log_eps}")

        q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            self._freeze_target(q_target)
        self._q_params = tuple(self.head.q.parameters())

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """Return a readable optimizer learning rate.

        Parameters
        ----------
        opt : Any
            Optimizer instance. Supports wrapped layouts exposing
            ``optim.param_groups``.

        Returns
        -------
        float
            Learning rate from param-group 0, or NaN if unavailable.
        """
        if hasattr(opt, "optim") and hasattr(opt.optim, "param_groups"):
            return float(opt.optim.param_groups[0]["lr"])
        if hasattr(opt, "param_groups"):
            return float(opt.param_groups[0]["lr"])
        return float("nan")

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Perform one C51 gradient update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch with fields:
            ``observations``, ``actions``, ``rewards``, ``next_observations``,
            and ``dones``. Optional PER fields are consumed via
            :func:`_get_per_weights`.

        Returns
        -------
        Dict[str, Any]
            Metrics dictionary with scalar logs and ``per/td_errors`` numpy array
            for priority updates.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        dist_all = self.head.q.dist(obs)  # (B, A, K)
        if dist_all.dim() != 3:
            raise ValueError(f"head.q.dist(obs) must be (B,A,K), got {tuple(dist_all.shape)}")
        n_atoms = int(dist_all.shape[-1])

        dist_a = dist_all.gather(1, act.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)  # (B, K)

        with th.no_grad():
            if self.double_dqn:
                q_next_online = self.head.q_values(nxt)  # (B, A)
                a_star = th.argmax(q_next_online, dim=-1)
            else:
                q_next_target = self.head.q_values_target(nxt)  # (B, A)
                a_star = th.argmax(q_next_target, dim=-1)

            next_dist_all = self.head.q_target.dist(nxt)  # (B, A, K)
            if next_dist_all.dim() != 3:
                raise ValueError(
                    f"head.q_target.dist(nxt) must be (B,A,K), got {tuple(next_dist_all.shape)}"
                )
            if int(next_dist_all.shape[-1]) != n_atoms:
                raise ValueError(
                    f"Atom count mismatch: online K={n_atoms} vs target K={int(next_dist_all.shape[-1])}"
                )

            next_dist = next_dist_all.gather(1, a_star.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)

            target_dist = _distribution_projection(
                next_dist=next_dist,
                rewards=rew,
                dones=done,
                gamma=self.gamma,
                support=self.head.support,
                v_min=float(self.head.v_min),
                v_max=float(self.head.v_max),
            )

        logp = th.log(dist_a.clamp(min=self.log_eps))
        per_sample = -(target_dist * logp).sum(dim=-1)  # (B,)
        loss = per_sample.mean() if w is None else (per_sample.view(-1, 1) * w).mean()

        self.opt.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self._clip_params(self._q_params, max_grad_norm=self.max_grad_norm, optimizer=self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_params(self._q_params, max_grad_norm=self.max_grad_norm)
            self.opt.step()

        if self.sched is not None:
            self.sched.step()

        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        with th.no_grad():
            support = self.head.support.to(self.device)
            if support.dim() != 1 or int(support.shape[0]) != n_atoms:
                raise ValueError(f"head.support must be (K,), got {tuple(support.shape)} vs K={n_atoms}")
            q_taken = (dist_a * support.view(1, -1)).sum(dim=-1)
            td_abs = per_sample.detach().abs().clamp(min=self.per_eps)

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_taken.mean())),
            "target/mean": float(_to_scalar(target_dist.mean())),
            "lr": float(self._get_optimizer_lr(self.opt)),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize core state and C51-specific hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "double_dqn": bool(self.double_dqn),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
                "log_eps": float(self.log_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore core state and C51-specific hyperparameters.

        Parameters
        ----------
        state : Mapping[str, Any]
            State mapping previously produced by :meth:`state_dict`.
        """
        super().load_state_dict(state)
        if "gamma" in state:
            self.gamma = float(state["gamma"])
        if "target_update_interval" in state:
            self.target_update_interval = int(state["target_update_interval"])
        if "tau" in state:
            self.tau = float(state["tau"])
        if "double_dqn" in state:
            self.double_dqn = bool(state["double_dqn"])
        if "max_grad_norm" in state:
            self.max_grad_norm = float(state["max_grad_norm"])
        if "per_eps" in state:
            self.per_eps = float(state["per_eps"])
        if "log_eps" in state:
            self.log_eps = float(state["log_eps"])
