"""Optimization core for Rainbow (C51) discrete control.

This module implements distributional TD updates using categorical projection
onto a fixed support, with optional Double-DQN action selection and n-step
backups.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th

from rllib.model_free.common.policies.base_core import QLearningCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _distribution_projection, _get_per_weights


class RainbowCore(QLearningCore):
    """Rainbow C51 learner core.

    Notes
    -----
    The core expects a head with ``q``, ``q_target``, ``q_values``,
    ``q_values_target``, ``support``, ``v_min``, and ``v_max``.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        n_step: int = 1,
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
        """Initialize Rainbow optimization core.

        Parameters
        ----------
        head : Any
            Policy head providing online/target categorical Q-distributions and
            support metadata.
        gamma : float, default=0.99
            Discount factor in ``[0, 1)``.
        n_step : int, default=1
            Backup horizon.
        target_update_interval : int, default=1000
            Hard target update period when ``tau == 0``.
        tau : float, default=0.0
            Soft update factor in ``[0, 1]``.
        double_dqn : bool, default=True
            Whether to use online argmax with target evaluation.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold; non-positive disables clipping.
        use_amp : bool, default=False
            Enable mixed precision training.
        per_eps : float, default=1e-6
            Floor for returned PER priority proxies.
        log_eps : float, default=1e-6
            Probability clamp floor before log.
        optim_name : str, default="adamw"
            Optimizer name.
        lr : float, default=3e-4
            Learning rate.
        weight_decay : float, default=0.0
            Weight decay.
        sched_name : str, default="none"
            Scheduler name.
        total_steps : int, default=0
            Scheduler horizon.
        warmup_steps : int, default=0
            Scheduler warmup steps.
        min_lr_ratio : float, default=0.0
            Minimum LR ratio for supported schedulers.
        poly_power : float, default=1.0
            Polynomial scheduler exponent.
        step_size : int, default=1000
            Step scheduler interval.
        sched_gamma : float, default=0.99
            Multiplicative scheduler factor.
        milestones : Sequence[int], default=()
            Milestones for multi-step schedulers.

        Raises
        ------
        ValueError
            If hyperparameters are invalid.
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
        self.n_step = int(n_step)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)
        self.log_eps = float(log_eps)

        if self.n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}")
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

        self._q_params = tuple(self.head.q.parameters())
        q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            self._freeze_target(q_target)

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """Get first optimizer param-group learning rate.

        Parameters
        ----------
        opt : Any
            Torch optimizer.

        Returns
        -------
        float
            Learning rate from param-group ``0``.
        """
        groups = getattr(opt, "param_groups", None)
        if not groups:
            return float("nan")
        return float(groups[0].get("lr", float("nan")))

    def _get_nstep_fields(self, batch: Any) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Resolve reward/done/next_obs tensors for current n-step setting.

        Parameters
        ----------
        batch : Any
            Replay batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(rewards, dones, next_obs)`` on ``self.device``.
        """
        if self.n_step <= 1:
            return (
                batch.rewards.to(self.device),
                batch.dones.to(self.device),
                batch.next_observations.to(self.device),
            )

        r = getattr(batch, "n_step_returns", None)
        d = getattr(batch, "n_step_dones", None)
        ns = getattr(batch, "n_step_next_observations", None)
        if (r is not None) and (d is not None) and (ns is not None):
            return (r.to(self.device), d.to(self.device), ns.to(self.device))

        return (
            batch.rewards.to(self.device),
            batch.dones.to(self.device),
            batch.next_observations.to(self.device),
        )

    def _gamma_n(self) -> float:
        """Compute effective discount ``gamma**n_step``.

        Returns
        -------
        float
            Effective n-step discount.
        """
        return float(self.gamma) ** max(1, int(self.n_step))

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Run one Rainbow (C51) optimization step.

        Parameters
        ----------
        batch : Any
            Replay batch with transition fields and optional n-step/PER fields.

        Returns
        -------
        Dict[str, Any]
            Logging dictionary including ``per/td_errors``.

        Raises
        ------
        ValueError
            If distribution tensor shapes are invalid.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()

        rew, done, nxt = self._get_nstep_fields(batch)
        rew = _to_column(rew)
        done = _to_column(done)

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        dist_all = self.head.q.dist(obs)
        if dist_all.dim() != 3:
            raise ValueError(f"head.q.dist(obs) must be (B,A,K), got {tuple(dist_all.shape)}")
        n_atoms = int(dist_all.shape[-1])
        dist_a = dist_all.gather(1, act.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)

        with th.no_grad():
            if self.double_dqn:
                if hasattr(self.head.q, "reset_noise"):
                    self.head.q.reset_noise()
                a_star = th.argmax(self.head.q_values(nxt), dim=-1)
            else:
                a_star = th.argmax(self.head.q_values_target(nxt), dim=-1)

            next_dist_all = self.head.q_target.dist(nxt)
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
                gamma=self._gamma_n(),
                support=self.head.support,
                v_min=float(self.head.v_min),
                v_max=float(self.head.v_max),
            )

        logp = th.log(dist_a.clamp(min=self.log_eps))
        per_sample = -(target_dist * logp).sum(dim=-1)
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
            "lr": self._get_optimizer_lr(self.opt),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize Rainbow core state.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "n_step": int(self.n_step),
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
        """Load Rainbow core state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.
        """
        super().load_state_dict(state)
