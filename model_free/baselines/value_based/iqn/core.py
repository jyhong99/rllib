"""Optimization core for Implicit Quantile Networks (IQN).

This module implements the IQN quantile TD update with optional Double-DQN
action selection and prioritized replay weighting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th

from rllib.model_free.common.policies.base_core import QLearningCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _get_per_weights, _quantile_huber_loss


class IQNCore(QLearningCore):
    """IQN learner core.

    Notes
    -----
    This core optimizes the online quantile network and periodically updates the
    target network using hard updates (interval) or Polyak averaging (``tau``).
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        double_dqn: bool = True,
        n_quantile_samples: int = 32,
        n_target_quantile_samples: int = 32,
        n_eval_quantile_samples: int = 32,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        per_eps: float = 1e-6,
        optim_name: str = "adamw",
        lr: float = 1e-4,
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
        """Initialize IQN optimization core.

        Parameters
        ----------
        head : Any
            IQN head providing quantile and value helper methods.
        gamma : float, default=0.99
            Discount factor in ``[0, 1)``.
        target_update_interval : int, default=1000
            Hard target update period when ``tau == 0``.
        tau : float, default=0.0
            Soft target update factor in ``[0, 1]``.
        double_dqn : bool, default=True
            Whether to use Double-DQN action selection.
        n_quantile_samples : int, default=32
            Number of sampled taus for current-state quantiles.
        n_target_quantile_samples : int, default=32
            Number of sampled taus for target quantiles.
        n_eval_quantile_samples : int, default=32
            Number of sampled taus for greedy action selection.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold. Non-positive disables clipping.
        use_amp : bool, default=False
            Enable mixed precision training.
        per_eps : float, default=1e-6
            Minimum TD error returned for PER priority updates.
        optim_name : str, default="adamw"
            Optimizer name.
        lr : float, default=1e-4
            Optimizer learning rate.
        weight_decay : float, default=0.0
            Optimizer weight decay.
        sched_name : str, default="none"
            Scheduler name.
        total_steps : int, default=0
            Scheduler total horizon.
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
            Milestones for multi-step scheduler.

        Raises
        ------
        ValueError
            If arguments are outside valid ranges.
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
        self.n_quantile_samples = int(n_quantile_samples)
        self.n_target_quantile_samples = int(n_target_quantile_samples)
        self.n_eval_quantile_samples = int(n_eval_quantile_samples)
        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.n_quantile_samples <= 0:
            raise ValueError(f"n_quantile_samples must be > 0, got {self.n_quantile_samples}")
        if self.n_target_quantile_samples <= 0:
            raise ValueError(
                f"n_target_quantile_samples must be > 0, got {self.n_target_quantile_samples}"
            )
        if self.n_eval_quantile_samples <= 0:
            raise ValueError(f"n_eval_quantile_samples must be > 0, got {self.n_eval_quantile_samples}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")

        self._q_params = tuple(self.head.q.parameters())
        self.head.freeze_target(self.head.q_target)

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """Get first parameter-group learning rate.

        Parameters
        ----------
        opt : Any
            Torch optimizer instance.

        Returns
        -------
        float
            Learning rate from the first optimizer group.
        """
        groups = getattr(opt, "param_groups", None)
        if not groups:
            return float("nan")
        return float(groups[0].get("lr", float("nan")))

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Run one IQN update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch containing ``observations``, ``actions``, ``rewards``,
            ``next_observations`` and ``dones``. May include PER ``weights``.

        Returns
        -------
        Dict[str, Any]
            Logging dictionary with scalar losses/means and ``per/td_errors``.

        Raises
        ------
        ValueError
            If quantile tensor shapes are inconsistent.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        taus = self.head.sample_taus(batch_size=bsz, n_samples=self.n_quantile_samples)
        cur_all = self.head.quantiles(obs, taus=taus)
        if cur_all.dim() != 3:
            raise ValueError(f"head.quantiles(obs, taus) must be (B,N,A), got {tuple(cur_all.shape)}")

        n_quant = int(cur_all.shape[1])
        cur = cur_all.gather(2, act.view(-1, 1, 1).expand(-1, n_quant, 1)).squeeze(-1)

        with th.no_grad():
            if self.double_dqn:
                a_star = th.argmax(self.head.q_values(nxt, n_samples=self.n_eval_quantile_samples), dim=-1)
            else:
                a_star = th.argmax(self.head.q_values_target(nxt, n_samples=self.n_eval_quantile_samples), dim=-1)

            taus_target = self.head.sample_taus(batch_size=bsz, n_samples=self.n_target_quantile_samples)
            nxt_t_all = self.head.quantiles_target(nxt, taus=taus_target)
            nxt_t = nxt_t_all.gather(
                2,
                a_star.view(-1, 1, 1).expand(-1, self.n_target_quantile_samples, 1),
            ).squeeze(-1)

            target = (
                rew.expand(-1, self.n_target_quantile_samples)
                + self.gamma * (1.0 - done.expand(-1, self.n_target_quantile_samples)) * nxt_t
            )

        loss, td_error = _quantile_huber_loss(
            current_quantiles=cur,
            target_quantiles=target,
            cum_prob=taus.unsqueeze(-1),
            weights=w,
        )

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
            q_mean_taken = cur.mean(dim=1)
            td = td_error.detach().view(-1).abs().clamp(min=self.per_eps)

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_mean_taken.mean())),
            "target/mean": float(_to_scalar(target.mean())),
            "lr": self._get_optimizer_lr(self.opt),
            "per/td_errors": td.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize IQN core state.

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
                "n_quantile_samples": int(self.n_quantile_samples),
                "n_target_quantile_samples": int(self.n_target_quantile_samples),
                "n_eval_quantile_samples": int(self.n_eval_quantile_samples),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load IQN core state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.
        """
        super().load_state_dict(state)
