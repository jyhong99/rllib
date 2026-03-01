"""DDPG core built by specializing the TD3 optimization core.

This module intentionally reuses :class:`TD3Core` and configures it to recover
the original DDPG update rule:

- single critic (Q1 path of TD3)
- no target policy smoothing (``policy_noise=0`` and ``noise_clip=0``)
- actor update every critic step (``policy_delay=1``)
"""

from __future__ import annotations

from typing import Any, Sequence

from rllib.model_free.baselines.policy_based.off_policy.td3.core import TD3Core


class DDPGCore(TD3Core):
    """DDPG optimization core implemented via TD3Core reuse.

    Parameters
    ----------
    head : Any
        DDPG-compatible head object exposing actor/critic networks, target
        networks, and prediction methods consumed by :class:`TD3Core`.
    gamma : float, default=0.99
        Discount factor used in TD target computation.
    tau : float, default=0.005
        Polyak interpolation coefficient for target updates.
    target_update_interval : int, default=1
        Number of gradient updates between target-network updates.
    actor_optim_name : str, default="adamw"
        Actor optimizer name resolved by the shared optimizer builder.
    actor_lr : float, default=3e-4
        Actor learning rate.
    actor_weight_decay : float, default=0.0
        Actor optimizer weight decay.
    critic_optim_name : str, default="adamw"
        Critic optimizer name.
    critic_lr : float, default=3e-4
        Critic learning rate.
    critic_weight_decay : float, default=0.0
        Critic optimizer weight decay.
    actor_sched_name : str, default="none"
        Actor scheduler policy.
    critic_sched_name : str, default="none"
        Critic scheduler policy.
    total_steps : int, default=0
        Total planned training steps for schedulers that require horizon.
    warmup_steps : int, default=0
        Scheduler warmup steps.
    min_lr_ratio : float, default=0.0
        Lower-bound ratio for cosine/polynomial schedulers.
    poly_power : float, default=1.0
        Polynomial decay power.
    step_size : int, default=1000
        Step scheduler interval.
    sched_gamma : float, default=0.99
        Multiplicative scheduler decay factor.
    milestones : Sequence[int], default=()
        Multi-step scheduler milestone indices.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold. Non-positive disables clipping.
    use_amp : bool, default=False
        Whether to enable automatic mixed precision during optimization.

    Notes
    -----
    DDPG behavior is recovered by forwarding to :class:`TD3Core` with:
    ``policy_noise=0.0``, ``noise_clip=0.0``, and ``policy_delay=1``.
    """

    def __init__(
        self,
        *,
        head: Any,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
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
        """Initialize a DDPG core by configuring TD3Core in DDPG mode.

        Parameters
        ----------
        head : Any
            Algorithm head object.
        gamma : float
            Discount factor.
        tau : float
            Target update coefficient.
        target_update_interval : int
            Frequency of target updates.
        actor_optim_name : str
            Actor optimizer identifier.
        actor_lr : float
            Actor learning rate.
        actor_weight_decay : float
            Actor weight decay.
        critic_optim_name : str
            Critic optimizer identifier.
        critic_lr : float
            Critic learning rate.
        critic_weight_decay : float
            Critic weight decay.
        actor_sched_name : str
            Actor scheduler identifier.
        critic_sched_name : str
            Critic scheduler identifier.
        total_steps : int
            Total step budget for schedulers.
        warmup_steps : int
            Warmup steps for schedulers.
        min_lr_ratio : float
            Minimum LR ratio for supported schedulers.
        poly_power : float
            Polynomial schedule exponent.
        step_size : int
            Step LR interval.
        sched_gamma : float
            Multiplicative LR decay.
        milestones : Sequence[int]
            Multi-step milestones.
        max_grad_norm : float
            Gradient clipping threshold.
        use_amp : bool
            Mixed precision flag.

        Returns
        -------
        None
            This constructor initializes the parent optimizer/scheduler state.
        """
        super().__init__(
            head=head,
            gamma=float(gamma),
            tau=float(tau),
            policy_noise=0.0,
            noise_clip=0.0,
            policy_delay=1,
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
