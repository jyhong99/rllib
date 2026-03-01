"""Optimization core for offline TD3+BC.

This module extends TD3 update logic with behavior-cloning regularization on the
actor objective, following the TD3+BC formulation for offline RL.
"""


from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th
import torch.nn.functional as F

from rllib.model_free.baselines.policy_based.off_policy.td3.core import TD3Core
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _get_per_weights


class TD3BCCore(TD3Core):
    """TD3+BC update core.

    Reuses TD3 critic/target updates and replaces the delayed actor objective:

    ``L_actor = MSE(pi(s), a_dataset) - lambda * Q1(s, pi(s))``

    where ``lambda = alpha / mean(|Q1(s, pi(s))|)``.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        target_update_interval: int = 1,
        alpha: float = 2.5,
        lambda_eps: float = 1e-6,
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
    ) -> None:
        """Initialize TD3+BC optimization hyperparameters and inherited TD3 state.

        Parameters
        ----------
        head : Any
            TD3-compatible policy head with actor/critic/target modules.
        gamma : float, default=0.99
            Discount factor in Bellman targets.
        tau : float, default=0.005
            Polyak averaging factor for target updates.
        policy_noise : float, default=0.2
            Target-policy smoothing noise standard deviation.
        noise_clip : float, default=0.5
            Clipping range for target noise.
        policy_delay : int, default=2
            Actor update delay relative to critic updates.
        target_update_interval : int, default=1
            Interval (in update calls) for target-network updates.
        alpha : float, default=2.5
            TD3+BC coefficient used in adaptive ``lambda`` scaling.
        lambda_eps : float, default=1e-6
            Numerical stabilizer when dividing by ``mean(|Q|)``.
        actor_optim_name, critic_optim_name : str
            Optimizer names.
        actor_lr, critic_lr : float
            Optimizer learning rates.
        actor_weight_decay, critic_weight_decay : float
            Optimizer weight decay values.
        actor_sched_name, critic_sched_name : str
            Scheduler names for actor/critic optimizers.
        total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
            Shared scheduler configuration.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold; ``0`` disables clipping.
        use_amp : bool, default=False
            Mixed-precision toggle.

        Returns
        -------
        None
            Initializes TD3+BC-specific runtime parameters in place.

        Raises
        ------
        ValueError
            If ``alpha`` or ``lambda_eps`` are non-positive.
        """
        super().__init__(
            head=head,
            gamma=float(gamma),
            tau=float(tau),
            policy_noise=float(policy_noise),
            noise_clip=float(noise_clip),
            policy_delay=int(policy_delay),
            target_update_interval=int(target_update_interval),
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
            max_grad_norm=float(max_grad_norm),
            use_amp=bool(use_amp),
        )

        self.alpha = float(alpha)
        self.lambda_eps = float(lambda_eps)
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.lambda_eps <= 0.0:
            raise ValueError(f"lambda_eps must be > 0, got {self.lambda_eps}")

    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """Run one TD3+BC gradient update from an offline replay batch.

        Parameters
        ----------
        batch : Any
            Batch containing ``observations``, ``actions``, ``rewards``,
            ``next_observations``, and ``dones``; optional PER weights are used
            when present.

        Returns
        -------
        dict[str, float]
            Metric dictionary with critic/actor losses, Q statistics, current
            learning rates, actor-update indicator, and PER TD-error feedback.
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
            next_a = self.head.target_action(
                nxt,
                noise_std=float(self.policy_noise),
                noise_clip=float(self.noise_clip),
            )
            q1_t, q2_t = self.head.q_values_target(nxt, next_a)
            q_t = th.min(q1_t, q2_t)
            target_q = rew + self.gamma * (1.0 - done) * q_t

        def _critic_loss_and_td() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """Compute TD3 critic loss and per-sample TD diagnostics.

            Returns
            -------
            tuple of torch.Tensor
                ``(loss, td_abs, q1, q2)`` where ``loss`` is weighted critic MSE,
                ``td_abs`` is absolute clipped-double TD error, and ``q1``/``q2``
                are current critic predictions.
            """
            q1, q2 = self.head.q_values(obs, act)
            l1 = F.mse_loss(q1, target_q, reduction="none")
            l2 = F.mse_loss(q2, target_q, reduction="none")
            per_sample = l1 + l2

            td = th.min(q1, q2) - target_q
            td_abs = td.abs().detach().squeeze(1)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
            return loss, td_abs, q1, q2

        self.critic_opt.zero_grad(set_to_none=True)
        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            critic_loss, td_abs, q1_now, q2_now = _critic_loss_and_td()
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        did_actor = (self.policy_delay > 0) and (self.update_calls % self.policy_delay == 0)
        actor_loss_scalar = 0.0
        bc_loss_scalar = 0.0
        actor_q_scalar = 0.0
        lambda_scalar = 0.0

        if did_actor:

            def _actor_loss() -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
                """Compute TD3+BC actor objective and diagnostics.

                Returns
                -------
                tuple of torch.Tensor
                    ``(actor_loss, bc_loss, q_mean, lam)`` where:

                    - ``bc_loss`` is behavior cloning MSE on dataset actions.
                    - ``q_mean`` is mean ``Q1(s, pi(s))``.
                    - ``lam`` is adaptive TD3+BC scale ``alpha / mean(|Q1|)``.
                """
                act_fn = getattr(self.head.actor, "act", None)
                if callable(act_fn):
                    pi, _ = act_fn(obs, deterministic=True)
                else:
                    pi = self.head.actor(obs)

                q1_pi, _ = self.head.q_values(obs, pi)
                bc_loss = F.mse_loss(pi, act)

                q_abs_mean = q1_pi.abs().mean().detach().clamp(min=self.lambda_eps)
                lam = self.alpha / q_abs_mean

                actor_loss = bc_loss - lam * q1_pi.mean()
                return actor_loss, bc_loss, q1_pi.mean(), lam

            self.actor_opt.zero_grad(set_to_none=True)
            if self.use_amp:
                with th.cuda.amp.autocast(enabled=True):
                    actor_loss, bc_loss, actor_q_mean, lam = _actor_loss()
                self.scaler.scale(actor_loss).backward()
                self._clip_params(
                    self.head.actor.parameters(),
                    max_grad_norm=self.max_grad_norm,
                    optimizer=self.actor_opt,
                )
                self.scaler.step(self.actor_opt)
                self.scaler.update()
            else:
                actor_loss, bc_loss, actor_q_mean, lam = _actor_loss()
                actor_loss.backward()
                self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
                self.actor_opt.step()

            if self.actor_sched is not None:
                self.actor_sched.step()

            actor_loss_scalar = float(_to_scalar(actor_loss))
            bc_loss_scalar = float(_to_scalar(bc_loss))
            actor_q_scalar = float(_to_scalar(actor_q_mean))
            lambda_scalar = float(_to_scalar(lam))

            self._maybe_update_target(
                target=getattr(self.head, "actor_target", None),
                source=self.head.actor,
                interval=self.target_update_interval,
                tau=self.tau,
            )
            self._maybe_update_target(
                target=getattr(self.head, "critic_target", None),
                source=self.head.critic,
                interval=self.target_update_interval,
                tau=self.tau,
            )

        out: Dict[str, float] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(actor_loss_scalar),
            "loss/actor_bc": float(bc_loss_scalar),
            "actor/q_mean": float(actor_q_scalar),
            "td3bc/lambda": float(lambda_scalar),
            "q/q1_mean": float(_to_scalar(q1_now.mean())),
            "q/q2_mean": float(_to_scalar(q2_now.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "td3bc/did_actor_update": float(1.0 if did_actor else 0.0),
        }
        out["per/td_errors"] = td_abs.detach().cpu().numpy()  # type: ignore[assignment]
        return out

    def state_dict(self) -> Dict[str, Any]:
        """Serialize TD3+BC core state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Base TD3 core state plus TD3+BC-specific scalars ``alpha`` and
            ``lambda_eps``.
        """
        s = super().state_dict()
        s.update(
            {
                "alpha": float(self.alpha),
                "lambda_eps": float(self.lambda_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore TD3+BC core state from checkpoint mapping.

        Parameters
        ----------
        state : Mapping[str, Any]
            Serialized mapping previously produced by :meth:`state_dict`.

        Returns
        -------
        None
            Loads inherited TD3 state and TD3+BC-specific scalars in place.
        """
        super().load_state_dict(state)
        if "alpha" in state:
            self.alpha = float(state["alpha"])
        if "lambda_eps" in state:
            self.lambda_eps = float(state["lambda_eps"])
