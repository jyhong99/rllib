"""Optimization core for FQF.

This module implements the two coupled FQF updates:

- Quantile value update for the tau-conditioned Q-network.
- Fraction proposal update for adaptive quantile fractions.

It is designed to plug into the generic off-policy algorithm wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch as th

from rllib.model_free.common.optimizers.optimizer_builder import build_optimizer
from rllib.model_free.common.optimizers.scheduler_builder import build_scheduler
from rllib.model_free.common.policies.base_core import QLearningCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _get_per_weights, _quantile_huber_loss


class FQFCore(QLearningCore):
    """FQF learner core with quantile and fraction updates.

    Notes
    -----
    FQF optimizes two parameter groups with separate optimizers:

    - ``head.q`` with quantile regression TD targets.
    - ``head.fraction_net`` with the fraction proposal objective.

    Both updates are executed on each call to :meth:`update_from_batch`.
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
        fraction_entropy_coef: float = 0.0,
        fraction_optim_name: str = "adamw",
        fraction_lr: float = 2.5e-4,
        fraction_weight_decay: float = 0.0,
        fraction_sched_name: str = "none",
        optim_name: str = "adamw",
        lr: float = 5e-5,
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
        """Initialize the FQF core.

        Parameters
        ----------
        head : Any
            Head object expected to provide ``q``, ``q_target``,
            ``fraction_net``, ``propose_fractions``, ``quantiles``,
            ``quantiles_target``, ``q_values``, and ``q_values_target``.
        gamma : float, default=0.99
            Discount factor in ``[0, 1)``.
        target_update_interval : int, default=1000
            Hard target update period when ``tau == 0``.
        tau : float, default=0.0
            Soft target update coefficient in ``[0, 1]``.
        double_dqn : bool, default=True
            Whether to use Double-DQN action selection.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold. Non-positive disables clipping.
        use_amp : bool, default=False
            Enable mixed precision for quantile-network update.
        per_eps : float, default=1e-6
            Epsilon used when returning PER TD errors.
        fraction_entropy_coef : float, default=0.0
            Entropy regularization coefficient for fraction loss.
        fraction_optim_name : str, default="adamw"
            Optimizer name for fraction network.
        fraction_lr : float, default=2.5e-4
            Learning rate for fraction network.
        fraction_weight_decay : float, default=0.0
            Weight decay for fraction optimizer.
        fraction_sched_name : str, default="none"
            Scheduler name for fraction optimizer.
        optim_name : str, default="adamw"
            Optimizer name for quantile network.
        lr : float, default=5e-5
            Learning rate for quantile optimizer.
        weight_decay : float, default=0.0
            Weight decay for quantile optimizer.
        sched_name : str, default="none"
            Scheduler name for quantile optimizer.
        total_steps : int, default=0
            Total scheduler horizon.
        warmup_steps : int, default=0
            Scheduler warmup steps.
        min_lr_ratio : float, default=0.0
            Minimum LR ratio for supported schedulers.
        poly_power : float, default=1.0
            Power used by polynomial scheduler.
        step_size : int, default=1000
            Step-size scheduler interval.
        sched_gamma : float, default=0.99
            Multiplicative scheduler decay.
        milestones : Sequence[int], default=()
            Milestones for multi-step schedules.

        Raises
        ------
        ValueError
            If hyperparameters or required head attributes are invalid.
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
        self.fraction_entropy_coef = float(fraction_entropy_coef)

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
        if not hasattr(self.head, "fraction_net"):
            raise ValueError("FQFCore requires head.fraction_net")

        self.fraction_opt = build_optimizer(
            self.head.fraction_net.parameters(),
            name=str(fraction_optim_name),
            lr=float(fraction_lr),
            weight_decay=float(fraction_weight_decay),
        )
        self.fraction_sched = build_scheduler(
            self.fraction_opt,
            name=str(fraction_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        # Cache parameter tuples to avoid repeated generator traversal per step.
        self._q_params = tuple(self.head.q.parameters())
        self._fraction_params = tuple(self.head.fraction_net.parameters())

        self.head.freeze_target(self.head.q_target)

    @staticmethod
    def _get_optimizer_lr(opt: Any) -> float:
        """Get first param-group learning rate.

        Parameters
        ----------
        opt : Any
            Torch optimizer instance.

        Returns
        -------
        float
            Learning rate of the first parameter group.
        """
        groups = getattr(opt, "param_groups", None)
        if not groups:
            return float("nan")
        return float(groups[0].get("lr", float("nan")))

    def _fraction_loss(
        self,
        *,
        taus: th.Tensor,
        quantiles_at_taus: th.Tensor,
        quantiles_at_tau_hats: th.Tensor,
        weights: th.Tensor | None,
        entropy: th.Tensor,
    ) -> th.Tensor:
        """Compute fraction proposal objective.

        Parameters
        ----------
        taus : torch.Tensor
            Fraction boundaries with shape ``(B, N+1)``.
        quantiles_at_taus : torch.Tensor
            Quantile values at interior taus with shape ``(B, N-1)`` for
            selected actions.
        quantiles_at_tau_hats : torch.Tensor
            Quantile values at tau midpoints with shape ``(B, N)`` for selected
            actions.
        weights : torch.Tensor | None
            Optional PER importance weights of shape ``(B, 1)``.
        entropy : torch.Tensor
            Fraction distribution entropy with shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar fraction loss.
        """
        grad_term = 2.0 * quantiles_at_taus - quantiles_at_tau_hats[:, :-1] - quantiles_at_tau_hats[:, 1:]
        per_sample = (taus[:, 1:-1] * grad_term.detach()).sum(dim=-1)

        frac = (per_sample * weights.view(-1)).mean() if weights is not None else per_sample.mean()
        if self.fraction_entropy_coef > 0.0:
            frac = frac - self.fraction_entropy_coef * entropy.mean()
        return frac

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Run one FQF optimization step from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch with fields ``observations``, ``actions``, ``rewards``,
            ``next_observations``, and ``dones``. Optional ``weights`` may be
            provided for PER.

        Returns
        -------
        Dict[str, Any]
            Scalar logs and PER TD errors under ``"per/td_errors"``.

        Raises
        ------
        ValueError
            If quantile tensor shapes from the head are inconsistent.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        taus, tau_hats, frac_entropy = self.head.propose_fractions(obs)
        tau_hats_det = tau_hats.detach()

        cur_all = self.head.quantiles(obs, tau_hats=tau_hats_det)
        if cur_all.dim() != 3:
            raise ValueError(f"head.quantiles(obs, tau_hats) must be (B,N,A), got {tuple(cur_all.shape)}")

        n_quant = int(cur_all.shape[1])
        gather_idx = act.view(-1, 1, 1).expand(-1, n_quant, 1)
        cur = cur_all.gather(2, gather_idx).squeeze(-1)

        with th.no_grad():
            _, tau_hats_next, _ = self.head.propose_fractions(nxt)
            nxt_t_all = self.head.quantiles_target(nxt, tau_hats=tau_hats_next.detach())

            if self.double_dqn:
                a_star = th.argmax(self.head.q_values(nxt), dim=-1)
            else:
                a_star = th.argmax(self.head.q_values_target(nxt), dim=-1)

            nxt_idx = a_star.view(-1, 1, 1).expand(-1, n_quant, 1)
            nxt_t = nxt_t_all.gather(2, nxt_idx).squeeze(-1)
            target = rew.expand(-1, n_quant) + self.gamma * (1.0 - done.expand(-1, n_quant)) * nxt_t

            interior_taus = taus[:, 1:-1].detach()
            quantiles_tau = self.head.quantiles(obs, tau_hats=interior_taus)
            quantiles_tau = quantiles_tau.gather(2, act.view(-1, 1, 1).expand(-1, n_quant - 1, 1)).squeeze(-1)

        fraction_loss = self._fraction_loss(
            taus=taus,
            quantiles_at_taus=quantiles_tau,
            quantiles_at_tau_hats=cur.detach(),
            weights=w,
            entropy=frac_entropy,
        )

        self.fraction_opt.zero_grad(set_to_none=True)
        fraction_loss.backward()
        self._clip_params(self._fraction_params, max_grad_norm=self.max_grad_norm)
        self.fraction_opt.step()
        if self.fraction_sched is not None:
            self.fraction_sched.step()

        quantile_loss, td_error = _quantile_huber_loss(
            current_quantiles=cur,
            target_quantiles=target,
            cum_prob=tau_hats_det.unsqueeze(-1),
            weights=w,
        )

        self.opt.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(quantile_loss).backward()
            self._clip_params(self._q_params, max_grad_norm=self.max_grad_norm, optimizer=self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            quantile_loss.backward()
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
            "loss/q": float(_to_scalar(quantile_loss)),
            "loss/fraction": float(_to_scalar(fraction_loss)),
            "q/mean": float(_to_scalar(q_mean_taken.mean())),
            "target/mean": float(_to_scalar(target.mean())),
            "fraction/entropy": float(_to_scalar(frac_entropy.mean())),
            "lr": self._get_optimizer_lr(self.opt),
            "lr/fraction": self._get_optimizer_lr(self.fraction_opt),
            "per/td_errors": td.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize core state including fraction optimizer/scheduler.

        Returns
        -------
        Dict[str, Any]
            Serializable state dictionary for checkpointing.
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
                "fraction_entropy_coef": float(self.fraction_entropy_coef),
                "fraction": self._save_opt_sched(self.fraction_opt, self.fraction_sched),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load core state and optional fraction optimizer/scheduler state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dictionary produced by :meth:`state_dict`.
        """
        super().load_state_dict(state)
        if "fraction" in state:
            self._load_opt_sched(self.fraction_opt, self.fraction_sched, state["fraction"])
