"""Factory entrypoint for offline TD3+BC.

The :func:`td3bc` function constructs a complete algorithm by composing:

- :class:`~rllib.model_free.baselines.offline.td3bc.head.TD3BCHead`
- :class:`~rllib.model_free.baselines.offline.td3bc.core.TD3BCCore`
- :class:`~rllib.model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`
"""


from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from rllib.model_free.baselines.offline.td3bc.core import TD3BCCore
from rllib.model_free.baselines.offline.td3bc.head import TD3BCHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def td3bc(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # ---------------------------------------------------------------------
    # Network (head)
    # ---------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    # ---------------------------------------------------------------------
    # TD3+BC update (core)
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    target_update_interval: int = 1,
    alpha: float = 2.5,
    lambda_eps: float = 1e-6,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # ---------------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------------
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) schedulers
    # ---------------------------------------------------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # ---------------------------------------------------------------------
    # OffPolicyAlgorithm replay/update schedule
    # ---------------------------------------------------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 0,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # ---------------------------------------------------------------------
    # PER
    # ---------------------------------------------------------------------
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_eps: float = 1e-6,
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
    use_her: bool = False,
    her_goal_shape: Any = None,
    her_reward_fn: Any = None,
    her_done_fn: Any = None,
    her_ratio: float = 0.8,
) -> OffPolicyAlgorithm:
    """Construct a configured TD3+BC offline algorithm instance.

    Parameters
    ----------
    obs_dim : int
        Flattened observation feature dimension.
    action_dim : int
        Continuous action dimension.
    device : str or torch.device, default="cpu"
        Device for module allocation and training.
    obs_shape : Any, optional
        Original observation shape for optional feature extractors.
    feature_extractor_cls : Any, optional
        Optional feature extractor class.
    feature_extractor_kwargs : Any, optional
        Keyword arguments for ``feature_extractor_cls``.
    init_trunk : Any, optional
        Optional trunk initialization control.
    hidden_sizes : tuple[int, ...], default=(256, 256)
        Hidden sizes used in actor/critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function for network blocks.
    init_type : str, default="orthogonal"
        Parameter initialization strategy.
    gain : float, default=1.0
        Gain parameter used by supported initializers.
    bias : float, default=0.0
        Bias initialization constant.
    action_low : np.ndarray, optional
        Lower action bounds.
    action_high : np.ndarray, optional
        Upper action bounds.
    gamma : float, default=0.99
        Discount factor.
    tau : float, default=0.005
        Polyak averaging factor for target updates.
    policy_noise : float, default=0.2
        Target-policy smoothing noise standard deviation.
    noise_clip : float, default=0.5
        Clip range for target-policy smoothing noise.
    policy_delay : int, default=2
        Delayed actor update interval.
    target_update_interval : int, default=1
        Interval for target parameter updates.
    alpha : float, default=2.5
        TD3+BC tradeoff coefficient controlling Q-term scaling.
    lambda_eps : float, default=1e-6
        Stabilizer used when computing ``lambda = alpha / mean(|Q|)``.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold; ``0`` disables clipping.
    use_amp : bool, default=False
        Mixed-precision update toggle.
    actor_optim_name, critic_optim_name : str
        Optimizer names for actor and critic.
    actor_lr, critic_lr : float
        Learning rates for actor and critic.
    actor_weight_decay, critic_weight_decay : float
        Weight decay values for actor and critic optimizers.
    actor_sched_name, critic_sched_name : str
        Scheduler names for actor and critic optimizers.
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Shared scheduler configuration.
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Minibatch size.
    update_after : int, default=0
        Number of environment steps before updates begin.
    update_every : int, default=1
        Number of environment steps between update calls.
    utd : float, default=1.0
        Update-to-data ratio used by the off-policy wrapper.
    gradient_steps : int, default=1
        Number of gradient updates per update call.
    max_updates_per_call : int, default=1000
        Safety cap on updates done in one call.
    use_per : bool, default=True
        Enable prioritized replay integration.
    per_alpha : float, default=0.6
        Prioritization exponent.
    per_beta : float, default=0.4
        Importance-sampling exponent.
    per_eps : float, default=1e-6
        Epsilon added to priorities.
    per_beta_final : float, default=1.0
        Final annealed PER beta.
    per_beta_anneal_steps : int, default=200000
        Number of steps to anneal PER beta.

    Returns
    -------
    OffPolicyAlgorithm
        Configured TD3+BC algorithm object.

    Raises
    ------
    ValueError
        If required shape/scheduling parameters are invalid.
    """
    obs_dim = int(obs_dim)
    action_dim = int(action_dim)
    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)

    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")
    if len(hidden_sizes_t) == 0 or any(h <= 0 for h in hidden_sizes_t):
        raise ValueError(f"hidden_sizes must be non-empty positive ints, got {hidden_sizes_t}")
    if int(buffer_size) <= 0:
        raise ValueError(f"buffer_size must be > 0, got {buffer_size}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if int(update_every) <= 0:
        raise ValueError(f"update_every must be > 0, got {update_every}")
    if int(gradient_steps) < 0:
        raise ValueError(f"gradient_steps must be >= 0, got {gradient_steps}")

    head = TD3BCHead(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        device=device,
        action_low=action_low,
        action_high=action_high,
        noise=None,
    )

    core = TD3BCCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        policy_noise=float(policy_noise),
        noise_clip=float(noise_clip),
        policy_delay=int(policy_delay),
        target_update_interval=int(target_update_interval),
        alpha=float(alpha),
        lambda_eps=float(lambda_eps),
        max_grad_norm=float(max_grad_norm),
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

    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=int(buffer_size),
        batch_size=int(batch_size),
        update_after=int(update_after),
        update_every=int(update_every),
        utd=float(utd),
        gradient_steps=int(gradient_steps),
        max_updates_per_call=int(max_updates_per_call),
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=int(per_beta_anneal_steps),
        use_her=bool(use_her),
        her_goal_shape=her_goal_shape,
        her_reward_fn=her_reward_fn,
        her_done_fn=her_done_fn,
        her_ratio=float(her_ratio),
    )
    return algo
