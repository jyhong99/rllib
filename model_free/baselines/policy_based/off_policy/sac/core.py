"""SAC optimization core with optional pixel regularization.

This module implements the update engine for continuous-action SAC and includes
optional DrQ/SVEA augmentation paths for pixel-based observations.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import torch as th
import torch.nn.functional as F

from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.policies.base_core import ActorCriticCore
from rllib.model_free.common.utils.policy_utils import _get_per_weights
from rllib.model_free.common.optimizers.optimizer_builder import build_optimizer
from rllib.model_free.common.optimizers.scheduler_builder import build_scheduler
from rllib.model_free.common.regularizations.drq import RandomShiftsAug
from rllib.model_free.common.regularizations.svea import RandomConvAug, svea_mix_loss


class SACCore(ActorCriticCore):
    """
    Soft Actor-Critic (SAC) update engine.

    This core implements the SAC learning rule for continuous-control settings while
    reusing :class:`~model_free.common.policies.base_core.ActorCriticCore` for shared
    training infrastructure (optimizers, schedulers, AMP, counters, and persistence).

    Overview
    --------
    SAC optimizes a maximum-entropy objective. Each update iteration typically performs:

    1. **Target construction (no grad)**

       .. math::

          y = r + \\gamma (1-d) \\left( \\min(Q_1^t(s', a'), Q_2^t(s', a')) - \\alpha \\log \\pi(a'|s') \\right)

       where :math:`a' \\sim \\pi(\\cdot|s')`.

    2. **Critic update (twin Q regression)**

       .. math::

          J_Q = \\mathbb{E}\\left[ (Q_1(s,a) - y)^2 + (Q_2(s,a) - y)^2 \\right]

       If PER is enabled, this regression may be importance-weighted.

    3. **Actor update (entropy-regularized)**

       .. math::

          J_\\pi = \\mathbb{E}\\left[ \\alpha \\log \\pi(a|s) - \\min(Q_1(s,a), Q_2(s,a)) \\right]

       with :math:`a \\sim \\pi(\\cdot|s)`.

    4. **Temperature update (optional; auto-alpha)**

       .. math::

          J_\\alpha = -\\mathbb{E}\\left[ \\log \\alpha \\, (\\log \\pi(a|s) + \\mathcal{H}_{\\text{target}}) \\right]

       This pushes entropy toward a target value.

    5. **Target critic update (Polyak / hard)** driven by the base core helper.

    Expected head interface (duck-typed)
    ------------------------------------
    Required attributes (discovered by :class:`ActorCriticCore`)
    - ``head.actor`` : torch.nn.Module
    - ``head.critic`` : torch.nn.Module
        Must support ``critic(obs, act) -> (q1, q2)`` with each output shaped ``(B, 1)``.
    - ``head.critic_target`` : torch.nn.Module
        Same signature as ``head.critic``.
    - ``head.device`` : torch.device or str (base core normalizes to ``self.device``).

    Required methods (used directly by this core)
    - ``head.sample_action_and_logp(obs) -> (action, logp)``
        - ``action``: tensor of shape ``(B, action_dim)``
        - ``logp``: tensor of shape ``(B, 1)`` preferred (``(B,)`` accepted and normalized)

    Shape conventions
    -----------------
    - Rewards and done flags are normalized to ``(B, 1)`` via :func:`_to_column`.
    - Log-probabilities are normalized to ``(B, 1)`` via :func:`_to_column`.

    Notes
    -----
    - This core returns ``"per/td_errors"`` as a NumPy array (non-scalar) to support
      PER priority updates in an outer algorithm wrapper. If you want strict typing,
      use ``Dict[str, Any]`` as the return type or omit the array and log only scalar
      aggregates.
    """

    def __init__(
        self,
        *,
        head: Any,
        # SAC hyperparameters
        gamma: float = 0.99,
        # target update
        tau: float = 0.005,
        target_update_interval: int = 1,
        # entropy temperature
        auto_alpha: bool = True,
        alpha_init: float = 0.2,
        target_entropy: Optional[float] = None,
        # alpha optimizer/scheduler
        alpha_optim_name: str = "adamw",
        alpha_lr: float = 3e-4,
        alpha_weight_decay: float = 0.0,
        alpha_sched_name: str = "none",
        # sched shared knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
        # grad / amp
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # actor/critic opt/sched (built by ActorCriticCore)
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        # pixel regularization
        pixel_regularization_mode: str = "off",  # off | drq | svea
        pixel_regularization_on: bool = True,
        pixel_skip_non_image: bool = True,
        drq_pad: int = 4,
        svea_kernel_size: int = 3,
        svea_alpha: float = 0.5,
        svea_beta: float = 0.5,
    ) -> None:
        """Initialize SAC update engine state and optimization components.

        Parameters
        ----------
        head : Any
            Head object exposing actor/critic/target modules and SAC forward helpers.
        gamma : float, default=0.99
            Discount factor for TD target construction.
        tau : float, default=0.005
            Polyak coefficient for target-network updates.
        target_update_interval : int, default=1
            Number of updates between target-network sync operations.
        auto_alpha : bool, default=True
            If True, optimize entropy temperature automatically.
        alpha_init : float, default=0.2
            Initial temperature value (stored as ``log_alpha``).
        target_entropy : float, optional
            Desired policy entropy; if None uses action-dimension heuristic.
        alpha_optim_name : str, default="adamw"
            Optimizer name for ``log_alpha``.
        alpha_lr : float, default=3e-4
            Learning rate for temperature optimizer.
        alpha_weight_decay : float, default=0.0
            Weight decay for temperature optimizer.
        alpha_sched_name : str, default="none"
            Scheduler name for temperature optimizer.
        total_steps : int, default=0
            Total training steps used by scheduler builders.
        warmup_steps : int, default=0
            Warmup steps used by scheduler builders.
        min_lr_ratio : float, default=0.0
            Minimum LR ratio for supported schedulers.
        poly_power : float, default=1.0
            Polynomial scheduler power.
        step_size : int, default=1000
            Step scheduler interval.
        sched_gamma : float, default=0.99
            Multiplicative scheduler factor.
        milestones : Sequence[int], default=()
            Multi-step scheduler milestones.
        max_grad_norm : float, default=0.0
            Gradient clipping threshold; non-positive disables clipping.
        use_amp : bool, default=False
            Enable mixed-precision autocast/scaler flow.
        actor_optim_name, critic_optim_name : str
            Optimizer names for actor and critic.
        actor_lr, critic_lr : float
            Learning rates for actor and critic optimizers.
        actor_weight_decay, critic_weight_decay : float
            Weight decay values for actor and critic optimizers.
        actor_sched_name, critic_sched_name : str
            Scheduler names for actor and critic optimizers.
        pixel_regularization_mode : str, default="off"
            Pixel regularization mode. One of ``{"off", "drq", "svea"}``.
        pixel_regularization_on : bool, default=True
            Master toggle for pixel regularization application.
        pixel_skip_non_image : bool, default=True
            If True, silently skip pixel regularization for non-image tensors.
            If False, raise when non-image data is passed while mode is active.
        drq_pad : int, default=4
            Padding size for random-shift augmentation in DrQ mode.
        svea_kernel_size : int, default=3
            Random convolution kernel size for SVEA mode (must be odd positive).
        svea_alpha : float, default=0.5
            SVEA mixing coefficient for original-sample loss component.
        svea_beta : float, default=0.5
            SVEA mixing coefficient for augmented-sample loss component.

        Returns
        -------
        None
            Constructor initializes optimizers, schedulers, and hyperparameters.

        Raises
        ------
        ValueError
            If any validated hyperparameter is outside supported ranges.
        """
        # ---------------------------------------------------------------------
        # Build base training infrastructure (actor/critic optimizers/schedulers)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Hyperparameters + validation
        # ---------------------------------------------------------------------
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)
        self.pixel_regularization_mode = str(pixel_regularization_mode).lower()
        self.pixel_regularization_on = bool(pixel_regularization_on)
        self.pixel_skip_non_image = bool(pixel_skip_non_image)
        self.drq_pad = int(drq_pad)
        self.svea_kernel_size = int(svea_kernel_size)
        self.svea_alpha = float(svea_alpha)
        self.svea_beta = float(svea_beta)

        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.pixel_regularization_mode not in {"off", "drq", "svea"}:
            raise ValueError(
                f"pixel_regularization_mode must be one of ['off', 'drq', 'svea'], got {self.pixel_regularization_mode}"
            )
        if self.drq_pad < 0:
            raise ValueError(f"drq_pad must be >= 0, got {self.drq_pad}")
        if self.svea_kernel_size <= 0 or self.svea_kernel_size % 2 == 0:
            raise ValueError(f"svea_kernel_size must be positive odd integer, got {self.svea_kernel_size}")

        # ---------------------------------------------------------------------
        # Target entropy
        # ---------------------------------------------------------------------
        # Common SAC heuristic: target_entropy = -|A| (continuous action dimension).
        if target_entropy is None:
            action_dim = int(getattr(self.head, "action_dim"))
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # ---------------------------------------------------------------------
        # Temperature parameter alpha (optimize log_alpha)
        # ---------------------------------------------------------------------
        self.auto_alpha = bool(auto_alpha)
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")

        self.log_alpha = th.tensor(
            float(math.log(float(alpha_init))),
            device=self.device,
            requires_grad=self.auto_alpha,
        )
        self.drq_aug = RandomShiftsAug(pad=self.drq_pad)
        self.svea_aug = RandomConvAug(kernel_size=self.svea_kernel_size)

        # alpha optimizer/scheduler (separate from ActorCriticCore)
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

        # Enforce target critic to be frozen (no gradients into target params).
        self.head.freeze_target(self.head.critic_target)

    # =============================================================================
    # Properties
    # =============================================================================
    @property
    def alpha(self) -> th.Tensor:
        """
        Current entropy temperature :math:`\\alpha = \\exp(\\log \\alpha)`.

        Returns
        -------
        torch.Tensor
            Scalar tensor on ``self.device``.
        """
        return self.log_alpha.exp()

    def _can_apply_pixel_reg(self, obs: th.Tensor, nxt: th.Tensor) -> bool:
        if not self.pixel_regularization_on or self.pixel_regularization_mode == "off":
            return False
        is_image = obs.ndim == 4 and nxt.ndim == 4
        if is_image:
            return True
        if self.pixel_skip_non_image:
            return False
        raise ValueError(
            "pixel regularization expects image tensors with shape (B,C,H,W), "
            f"got obs={tuple(obs.shape)} nxt={tuple(nxt.shape)}"
        )

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one SAC update from a sampled replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch object (duck-typed). Expected fields:

            - ``observations`` : torch.Tensor, shape (B, obs_dim)
            - ``actions`` : torch.Tensor, shape (B, action_dim)
            - ``rewards`` : torch.Tensor, shape (B,) or (B, 1)
            - ``next_observations`` : torch.Tensor, shape (B, obs_dim)
            - ``dones`` : torch.Tensor, shape (B,) or (B, 1)

            Optional PER fields may be present and will be interpreted by
            :func:`~model_free.common.utils.policy_utils._get_per_weights`.

        Returns
        -------
        metrics : Dict[str, Any]
            Logging metrics. All values are scalars except:

            - ``"per/td_errors"`` : np.ndarray, shape (B,)
              Absolute TD-error proxy for PER priority updates upstream.

        Notes
        -----
        - This method increments internal update counters via ``self._bump()``.
        - AMP behavior is controlled by ``self.use_amp`` from the base core.
        """
        self._bump()

        # ---------------------------------------------------------------------
        # Move batch to device + normalize shapes
        # ---------------------------------------------------------------------
        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)
        rew = _to_column(batch.rewards.to(self.device))            # (B,1)
        nxt = batch.next_observations.to(self.device)
        done = _to_column(batch.dones.to(self.device))             # (B,1)

        use_pixel_reg = self._can_apply_pixel_reg(obs, nxt)
        mode = self.pixel_regularization_mode if use_pixel_reg else "off"

        if mode == "drq":
            obs_aug = self.drq_aug(obs)
            nxt_aug = self.drq_aug(nxt)
        elif mode == "svea":
            obs_aug = self.svea_aug(obs)
            nxt_aug = self.svea_aug(nxt)
        else:
            obs_aug = obs
            nxt_aug = nxt

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)         # (B,1) or None

        def _build_target(nxt_in: th.Tensor) -> th.Tensor:
            with th.no_grad():
                nxt_a, nxt_logp = self.head.sample_action_and_logp(nxt_in)
                nxt_logp = _to_column(nxt_logp)
                q1_t, q2_t = self.head.q_values_target(nxt_in, nxt_a)
                q_min_t = th.min(q1_t, q2_t)
                return rew + self.gamma * (1.0 - done) * (q_min_t - self.alpha * nxt_logp)

        target_q = _build_target(nxt)
        target_q_aug = _build_target(nxt_aug) if mode == "svea" else target_q

        # ---------------------------------------------------------------------
        # Critic update
        # ---------------------------------------------------------------------
        def _critic_loss_and_td(obs_in: th.Tensor, tgt_in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
            """
            Compute critic regression loss and a TD-error magnitude proxy.

            Returns
            -------
            loss : torch.Tensor
                Scalar loss tensor.
            td_abs : torch.Tensor
                Shape (B,) absolute TD error proxy suitable for PER priorities.
            """
            q1, q2 = self.head.q_values(obs_in, act)               # each (B,1)

            per_sample = (
                F.mse_loss(q1, tgt_in, reduction="none")
                + F.mse_loss(q2, tgt_in, reduction="none")
            )                                                      # (B,1)

            # TD magnitude proxy (min(Q1,Q2) matches the target structure)
            td = th.min(q1, q2) - tgt_in
            td_abs = td.abs().detach().squeeze(1)                  # (B,)

            loss = per_sample.mean() if w is None else (w * per_sample).mean()
            return loss, td_abs

        self.critic_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                critic_loss, td_abs = _critic_loss_and_td(obs_aug if mode == "drq" else obs, target_q)
                if mode == "svea":
                    critic_loss_aug, _ = _critic_loss_and_td(obs_aug, target_q_aug)
                    critic_loss = svea_mix_loss(
                        critic_loss,
                        critic_loss_aug,
                        alpha=self.svea_alpha,
                        beta=self.svea_beta,
                    )
            self.scaler.scale(critic_loss).backward()
            self._clip_params(
                self.head.critic.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.critic_opt,
            )
            self.scaler.step(self.critic_opt)
        else:
            critic_loss, td_abs = _critic_loss_and_td(obs_aug if mode == "drq" else obs, target_q)
            if mode == "svea":
                critic_loss_aug, _ = _critic_loss_and_td(obs_aug, target_q_aug)
                critic_loss = svea_mix_loss(
                    critic_loss,
                    critic_loss_aug,
                    alpha=self.svea_alpha,
                    beta=self.svea_beta,
                )
            critic_loss.backward()
            self._clip_params(self.head.critic.parameters(), max_grad_norm=self.max_grad_norm)
            self.critic_opt.step()

        if self.critic_sched is not None:
            self.critic_sched.step()

        # ---------------------------------------------------------------------
        # Actor update
        # ---------------------------------------------------------------------
        def _actor_loss(obs_in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Compute SAC actor loss on the current batch.

            Returns
            -------
            loss : torch.Tensor
                Scalar actor loss.
            logp : torch.Tensor
                Shape (B,1) log π(a|s) for sampled actions.
            q_pi : torch.Tensor
                Shape (B,1) min(Q1,Q2)(s,a) for sampled actions.
            """
            new_a, logp = self.head.sample_action_and_logp(obs_in)
            logp = _to_column(logp)

            q1_pi, q2_pi = self.head.q_values(obs_in, new_a)
            q_pi = th.min(q1_pi, q2_pi)

            loss = (self.alpha * logp - q_pi).mean()
            return loss, logp, q_pi

        self.actor_opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                actor_loss, logp, q_pi = _actor_loss(obs_aug if mode == "drq" else obs)
                if mode == "svea":
                    actor_loss_aug, logp_aug, q_pi_aug = _actor_loss(obs_aug)
                    actor_loss = svea_mix_loss(
                        actor_loss,
                        actor_loss_aug,
                        alpha=self.svea_alpha,
                        beta=self.svea_beta,
                    )
                    logp = 0.5 * (logp + logp_aug)
                    q_pi = 0.5 * (q_pi + q_pi_aug)
            self.scaler.scale(actor_loss).backward()
            self._clip_params(
                self.head.actor.parameters(),
                max_grad_norm=self.max_grad_norm,
                optimizer=self.actor_opt,
            )
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss, logp, q_pi = _actor_loss(obs_aug if mode == "drq" else obs)
            if mode == "svea":
                actor_loss_aug, logp_aug, q_pi_aug = _actor_loss(obs_aug)
                actor_loss = svea_mix_loss(
                    actor_loss,
                    actor_loss_aug,
                    alpha=self.svea_alpha,
                    beta=self.svea_beta,
                )
                logp = 0.5 * (logp + logp_aug)
                q_pi = 0.5 * (q_pi + q_pi_aug)
            actor_loss.backward()
            self._clip_params(self.head.actor.parameters(), max_grad_norm=self.max_grad_norm)
            self.actor_opt.step()

        if self.actor_sched is not None:
            self.actor_sched.step()

        # ---------------------------------------------------------------------
        # Alpha update (optional)
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            # Detach logp to prevent alpha loss from backpropagating into the actor.
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if self.alpha_sched is not None:
                self.alpha_sched.step()

            alpha_loss_val = float(_to_scalar(alpha_loss))

        # ---------------------------------------------------------------------
        # Target critic update (Polyak / hard depending on core helper)
        # ---------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "critic_target", None),
            source=self.head.critic,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        with th.no_grad():
            q1_b, q2_b = self.head.critic(obs_aug if mode == "drq" else obs, act)

        out: Dict[str, Any] = {
            "loss/critic": float(_to_scalar(critic_loss)),
            "loss/actor": float(_to_scalar(actor_loss)),
            "loss/alpha": float(alpha_loss_val),
            "alpha": float(_to_scalar(self.alpha)),
            "q/q1_mean": float(_to_scalar(q1_b.mean())),
            "q/q2_mean": float(_to_scalar(q2_b.mean())),
            "q/pi_min_mean": float(_to_scalar(q_pi.mean())),
            "logp_mean": float(_to_scalar(logp.mean())),
            "lr/actor": float(self.actor_opt.param_groups[0]["lr"]),
            "lr/critic": float(self.critic_opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
            "pixel_reg/enabled": float(1.0 if use_pixel_reg else 0.0),
        }
        if use_pixel_reg:
            out["pixel_reg/mode"] = mode
        if self.alpha_opt is not None:
            out["lr/alpha"] = float(self.alpha_opt.param_groups[0]["lr"])

        return out

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state, including alpha state if auto-alpha is enabled.

        Returns
        -------
        Dict[str, Any]
            State dictionary including:
            - base core state (optimizers/schedulers/counters)
            - ``log_alpha`` and alpha optimizer/scheduler state (if present)
            - ``auto_alpha`` and ``target_entropy`` (for inspection/debugging)

        Notes
        -----
        Hyperparameters are typically constructor-owned; these fields are stored
        primarily for reproducibility and debugging.
        """
        s = super().state_dict()
        s.update(
            {
                "log_alpha": float(_to_scalar(self.log_alpha)),
                "alpha": self._save_opt_sched(self.alpha_opt, self.alpha_sched) if self.alpha_opt is not None else None,
                "auto_alpha": bool(self.auto_alpha),
                "target_entropy": float(self.target_entropy),
                "pixel_regularization_mode": str(self.pixel_regularization_mode),
                "pixel_regularization_on": bool(self.pixel_regularization_on),
                "pixel_skip_non_image": bool(self.pixel_skip_non_image),
                "drq_pad": int(self.drq_pad),
                "svea_kernel_size": int(self.svea_kernel_size),
                "svea_alpha": float(self.svea_alpha),
                "svea_beta": float(self.svea_beta),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state, including alpha optimizer/scheduler if present.

        Parameters
        ----------
        state : Mapping[str, Any]
            State payload produced by :meth:`state_dict`.

        Notes
        -----
        - This method assumes constructor configuration is compatible with the
          serialized state (e.g., auto_alpha enabled/disabled consistently).
        - Optimizer reconstruction is not performed here; it is expected to be
          done in ``__init__`` and then populated with loaded state.
        """
        super().load_state_dict(state)

        if "log_alpha" in state:
            with th.no_grad():
                self.log_alpha.copy_(th.tensor(float(state["log_alpha"]), device=self.device))

        # Best-effort restore of configuration metadata (does not rebuild optimizers)
        if "auto_alpha" in state:
            self.auto_alpha = bool(state["auto_alpha"])
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])
        if "pixel_regularization_mode" in state:
            self.pixel_regularization_mode = str(state["pixel_regularization_mode"]).lower()
        if "pixel_regularization_on" in state:
            self.pixel_regularization_on = bool(state["pixel_regularization_on"])
        if "pixel_skip_non_image" in state:
            self.pixel_skip_non_image = bool(state["pixel_skip_non_image"])
        if "drq_pad" in state:
            self.drq_pad = int(state["drq_pad"])
            self.drq_aug = RandomShiftsAug(pad=self.drq_pad)
        if "svea_kernel_size" in state:
            self.svea_kernel_size = int(state["svea_kernel_size"])
            self.svea_aug = RandomConvAug(kernel_size=self.svea_kernel_size)
        if "svea_alpha" in state:
            self.svea_alpha = float(state["svea_alpha"])
        if "svea_beta" in state:
            self.svea_beta = float(state["svea_beta"])

        alpha_state = state.get("alpha", None)
        if self.alpha_opt is not None and isinstance(alpha_state, Mapping):
            self._load_opt_sched(self.alpha_opt, self.alpha_sched, alpha_state)
