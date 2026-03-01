"""Optimization core for offline Implicit Q-Learning (IQL).

This module implements IQL update rules on top of common actor-critic utilities:

- Bellman regression for twin Q critics.
- Expectile regression for a separate state-value network.
- Advantage-weighted behavior cloning for the actor.
"""


from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from rllib.model_free.common.optimizers.optimizer_builder import build_optimizer
from rllib.model_free.common.optimizers.scheduler_builder import build_scheduler
from rllib.model_free.common.policies.base_core import ActorCriticCore
from rllib.model_free.common.utils.common_utils import _reduce_joint, _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _expectile_loss, _get_per_weights


class IQLCore(ActorCriticCore):
    """Training logic container for continuous-action IQL.

    Notes
    -----
    Replay interaction and update scheduling are handled by the shared
    ``OffPolicyAlgorithm`` wrapper; this class only defines gradient updates
    from already-sampled batches.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        expectile: float = 0.7,
        beta: float = 3.0,
        max_weight: float = 100.0,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        value_optim_name: str = "adamw",
        value_lr: float = 3e-4,
        value_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        value_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        """Initialize IQL optimization hyperparameters and optimizers.

        Parameters
        ----------
        head : Any
            IQL-compatible head exposing ``actor``, ``critic``, ``critic_target``,
            and ``value`` modules.
        gamma : float, default=0.99
            Discount factor used in critic target computation.
        tau : float, default=0.005
            Polyak coefficient for target critic updates.
        target_update_interval : int, default=1
            Number of updates between target updates.
        expectile : float, default=0.7
            Expectile parameter for value regression.
        beta : float, default=3.0
            Advantage temperature used by actor weighting ``exp(beta * adv)``.
        max_weight : float, default=100.0
            Maximum cap applied to actor sample weights.
        max_grad_norm : float, default=0.0
            Gradient clipping norm threshold; ``0`` disables clipping.
        use_amp : bool, default=False
            Mixed-precision toggle.
        actor_optim_name, actor_lr, actor_weight_decay
            Actor optimizer configuration.
        critic_optim_name, critic_lr, critic_weight_decay
            Critic optimizer configuration.
        value_optim_name, value_lr, value_weight_decay
            Value-network optimizer configuration.
        actor_sched_name, critic_sched_name, value_sched_name : str
            Scheduler names for each optimizer.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Shared scheduler hyperparameters.

        Returns
        -------
        None
            Initializes optimizers, schedulers, and runtime hyperparameters.

        Raises
        ------
        ValueError
            If key hyperparameters are out of range or required head members are
            missing.
        """
        super().__init__(
            head=head,
            use_amp=use_amp,
            actor_optim_name=actor_optim_name,
            actor_lr=actor_lr,
            actor_weight_decay=actor_weight_decay,
            critic_optim_name=critic_optim_name,
            critic_lr=critic_lr,
            critic_weight_decay=critic_weight_decay,
            actor_sched_name=actor_sched_name,
            critic_sched_name=critic_sched_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            poly_power=poly_power,
            step_size=step_size,
            sched_gamma=sched_gamma,
            milestones=milestones,
        )

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.expectile = float(expectile)
        self.beta = float(beta)
        self.max_weight = float(max_weight)
        self.max_grad_norm = float(max_grad_norm)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 < self.expectile < 1.0):
            raise ValueError(f"expectile must be in (0,1), got {self.expectile}")
        if self.max_weight <= 0.0:
            raise ValueError(f"max_weight must be > 0, got {self.max_weight}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        if not hasattr(self.head, "value"):
            raise ValueError("IQLCore requires head.value network")

        self.value_opt = build_optimizer(
            self.head.value.parameters(),
            name=str(value_optim_name),
            lr=float(value_lr),
            weight_decay=float(value_weight_decay),
        )
        self.value_sched = build_scheduler(
            self.value_opt,
            name=str(value_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

        self.head.freeze_target(self.head.critic_target)

    def _critic_loss(
        self,
        *,
        obs: th.Tensor,
        act: th.Tensor,
        rew: th.Tensor,
        nxt: th.Tensor,
        done: th.Tensor,
        w: th.Tensor | None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute twin-critic Bellman regression loss.

        Parameters
        ----------
        obs : torch.Tensor
            Current observations, shape ``(B, obs_dim)``.
        act : torch.Tensor
            Dataset actions, shape ``(B, action_dim)``.
        rew : torch.Tensor
            Rewards, shape ``(B, 1)``.
        nxt : torch.Tensor
            Next observations, shape ``(B, obs_dim)``.
        done : torch.Tensor
            Episode termination flags, shape ``(B, 1)``.
        w : torch.Tensor or None
            Optional per-sample PER importance weights.

        Returns
        -------
        tuple of torch.Tensor
            ``(loss, td_abs, q_mean)`` where ``loss`` is the critic objective,
            ``td_abs`` are absolute TD errors per sample, and ``q_mean`` is the
            detached minibatch mean of ``min(Q1, Q2)``.
        """
        with th.no_grad():
            next_v = _to_column(self.head.value_values(nxt))
            target_q = rew + self.gamma * (1.0 - done) * next_v

        q1, q2 = self.head.q_values(obs, act)
        per_sample = F.mse_loss(q1, target_q, reduction="none") + F.mse_loss(q2, target_q, reduction="none")
        loss = per_sample.mean() if w is None else (w * per_sample).mean()

        td_abs = (th.min(q1, q2) - target_q).abs().detach().squeeze(1)
        q_mean = th.min(q1, q2).detach().mean()
        return loss, td_abs, q_mean

    def _value_loss(
        self,
        *,
        obs: th.Tensor,
        act: th.Tensor,
        w: th.Tensor | None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute expectile value-network regression loss.

        Parameters
        ----------
        obs : torch.Tensor
            Current observations, shape ``(B, obs_dim)``.
        act : torch.Tensor
            Dataset actions, shape ``(B, action_dim)``.
        w : torch.Tensor or None
            Optional per-sample PER importance weights.

        Returns
        -------
        tuple of torch.Tensor
            ``(loss, v_mean, adv_mean)`` where ``loss`` is expectile regression
            loss, ``v_mean`` is detached mean state value, and ``adv_mean`` is
            detached mean of ``Q - V``.
        """
        with th.no_grad():
            q1, q2 = self.head.q_values(obs, act)
            q = th.min(q1, q2)

        v = _to_column(self.head.value_values(obs))
        diff = q - v
        loss = _expectile_loss(diff, expectile=self.expectile, weights=w)
        return loss, v.detach().mean(), diff.detach().mean()

    def _actor_loss(
        self,
        *,
        obs: th.Tensor,
        act: th.Tensor,
        w: th.Tensor | None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute advantage-weighted actor objective.

        Parameters
        ----------
        obs : torch.Tensor
            Current observations, shape ``(B, obs_dim)``.
        act : torch.Tensor
            Dataset actions, shape ``(B, action_dim)``.
        w : torch.Tensor or None
            Optional per-sample PER importance weights.

        Returns
        -------
        tuple of torch.Tensor
            ``(loss, weight_mean, adv_mean)`` where ``loss`` is negative weighted
            log-likelihood of dataset actions, ``weight_mean`` is detached mean
            exp-advantage weight, and ``adv_mean`` is detached mean advantage.
        """
        dist = self.head.actor.get_dist(obs)
        logp = _to_column(_reduce_joint(dist.log_prob(act)))

        with th.no_grad():
            q1, q2 = self.head.q_values(obs, act)
            q = th.min(q1, q2)
            v = _to_column(self.head.value_values(obs))
            adv = q - v
            weight = th.exp(self.beta * adv).clamp(max=self.max_weight)

        per_sample = -(weight * logp)
        loss = per_sample.mean() if w is None else (w * per_sample).mean()
        return loss, weight.detach().mean(), adv.detach().mean()

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """Run one complete IQL update from a replay batch.

        Parameters
        ----------
        batch : Any
            Batch containing at least ``observations``, ``actions``, ``rewards``,
            ``next_observations``, and ``dones``. PER fields are used when present.

        Returns
        -------
        dict[str, Any]
            Scalar metrics and ``per/td_errors`` for priority updates.

        Notes
        -----
        Update order is critic -> value -> actor, followed by optional scheduler
        steps and target critic Polyak update.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))

        bsz = int(obs.shape[0])
        w = _get_per_weights(batch, bsz, device=self.device)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss, td_abs, q_mean = self._critic_loss(obs=obs, act=act, rew=rew, nxt=nxt, done=done, w=w)
        critic_loss.backward()
        self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
        self.critic_opt.step()

        self.value_opt.zero_grad(set_to_none=True)
        value_loss, v_mean, adv_mean = self._value_loss(obs=obs, act=act, w=w)
        value_loss.backward()
        self._clip_params(self.head.value.parameters(), max_grad_norm=self.max_grad_norm)
        self.value_opt.step()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss, weight_mean, actor_adv_mean = self._actor_loss(obs=obs, act=act, w=w)
        actor_loss.backward()
        self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
        self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()
        if self.value_sched is not None:
            self.value_sched.step()

        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        return {
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/value": float(_to_scalar(value_loss)),
            "q/mean": float(_to_scalar(q_mean)),
            "v/mean": float(_to_scalar(v_mean)),
            "adv/mean": float(_to_scalar(adv_mean)),
            "actor_adv/mean": float(_to_scalar(actor_adv_mean)),
            "weights/mean": float(_to_scalar(weight_mean)),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "lr/value": float(self.value_opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialize IQL core state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Base core state plus IQL-specific hyperparameters and value optimizer
            scheduler state.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "tau": float(self.tau),
                "target_update_interval": int(self.target_update_interval),
                "expectile": float(self.expectile),
                "beta": float(self.beta),
                "max_weight": float(self.max_weight),
                "max_grad_norm": float(self.max_grad_norm),
                "value": self._save_opt_sched(self.value_opt, self.value_sched),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore IQL core state from checkpoint payload.

        Parameters
        ----------
        state : Mapping[str, Any]
            Mapping produced by :meth:`state_dict`.

        Returns
        -------
        None
            Loads optimizer/scheduler and runtime parameters in place.
        """
        super().load_state_dict(state)
        if "value" in state:
            self._load_opt_sched(self.value_opt, self.value_sched, state["value"])
        return
