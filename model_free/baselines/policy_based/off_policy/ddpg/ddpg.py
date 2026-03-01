"""DDPG algorithm builder.

This module exposes a high-level factory that assembles:

1. :class:`DDPGHead` (networks + target networks + action noise handling),
2. :class:`DDPGCore` (optimization logic),
3. :class:`OffPolicyAlgorithm` (replay scheduling + update orchestration).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from rllib.model_free.common.noises.noise_builder import build_noise
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm

from rllib.model_free.baselines.policy_based.off_policy.ddpg.core import DDPGCore
from rllib.model_free.baselines.policy_based.off_policy.ddpg.head import DDPGHead


def ddpg(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # network
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    # exploration noise (head-owned)
    exploration_noise: Optional[str] = None,
    noise_mu: float = 0.0,
    noise_sigma: float = 0.1,
    ou_theta: float = 0.15,
    ou_dt: float = 1e-2,
    uniform_low: float = -1.0,
    uniform_high: float = 1.0,
    action_noise_eps: float = 1e-6,
    action_noise_low: Optional[Union[float, Sequence[float]]] = None,
    action_noise_high: Optional[Union[float, Sequence[float]]] = None,
    noise_clip: Optional[float] = None,
    # ddpg core
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # optimizers
    actor_optim_name: str = "adamw",
    actor_lr: float = 3e-4,
    actor_weight_decay: float = 0.0,
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    # schedulers
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
    # replay/schedule
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # PER
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
    """Build a DDPG off-policy algorithm object.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension used by MLP-based actor/critic inputs.
    action_dim : int
        Continuous action dimension.
    device : Union[str, th.device], default="cpu"
        Compute device used by model modules and optimization.
    obs_shape : Any, optional
        Optional raw observation shape for feature extractors.
    feature_extractor_cls : Any, optional
        Optional feature extractor class/callable used before MLP trunk.
    feature_extractor_kwargs : Any, optional
        Keyword arguments for feature extractor construction.
    init_trunk : Any, optional
        Optional trunk-initialization override passed to network builder.
    hidden_sizes : tuple[int, ...], default=(256, 256)
        Shared hidden layer sizes for actor and critic MLPs.
    activation_fn : Any, default=th.nn.ReLU
        Activation class/function used in actor/critic MLPs.
    init_type : str, default="orthogonal"
        Weight initialization strategy for newly created modules.
    gain : float, default=1.0
        Gain passed to initialization utility where applicable.
    bias : float, default=0.0
        Bias initialization value.
    action_low, action_high : np.ndarray, optional
        Optional action bounds consumed by deterministic actor squashing/rescaling.
    exploration_noise : str, optional
        Action noise kind consumed by :func:`build_noise` (e.g., OU/Gaussian).
    noise_mu : float, default=0.0
        Mean for supported noise processes.
    noise_sigma : float, default=0.1
        Standard deviation for supported noise processes.
    ou_theta : float, default=0.15
        OU-process theta.
    ou_dt : float, default=1e-2
        OU-process delta-time.
    uniform_low, uniform_high : float
        Uniform-noise range.
    action_noise_eps : float, default=1e-6
        Numerical epsilon used by action-noise utilities.
    action_noise_low, action_noise_high : float or Sequence[float], optional
        Optional clipping bounds for sampled exploration noise.
    noise_clip : float, optional
        Optional clipping bound forwarded to head-level action-noise path.
    gamma : float, default=0.99
        Discount factor.
    tau : float, default=0.005
        Polyak target update coefficient.
    target_update_interval : int, default=1
        Number of gradient steps per target network update.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold; non-positive disables clipping.
    use_amp : bool, default=False
        Enable automatic mixed precision.
    actor_optim_name, critic_optim_name : str
        Optimizer identifiers for actor and critic.
    actor_lr, critic_lr : float
        Learning rates for actor and critic.
    actor_weight_decay, critic_weight_decay : float
        Weight decay for actor and critic optimizers.
    actor_sched_name, critic_sched_name : str
        Scheduler identifiers.
    total_steps : int, default=0
        Total training steps for schedulers requiring horizon.
    warmup_steps : int, default=0
        Scheduler warmup steps.
    min_lr_ratio : float, default=0.0
        Minimum learning-rate ratio for supported schedulers.
    poly_power : float, default=1.0
        Polynomial scheduler power.
    step_size : int, default=1000
        Step-based scheduler interval.
    sched_gamma : float, default=0.99
        Multiplicative scheduler factor.
    milestones : tuple[int, ...], default=()
        Multi-step scheduler milestones.
    buffer_size : int, default=1_000_000
        Replay capacity.
    batch_size : int, default=256
        Batch size sampled from replay for each gradient step.
    update_after : int, default=1_000
        Minimum environment steps before updates begin.
    update_every : int, default=1
        Update trigger cadence in environment steps.
    utd : float, default=1.0
        Update-to-data ratio hint passed to off-policy driver.
    gradient_steps : int, default=1
        Number of gradient steps per update trigger.
    max_updates_per_call : int, default=1_000
        Hard upper bound for updates in a single ``update()`` call.
    use_per : bool, default=True
        Whether prioritized replay is enabled.
    per_alpha : float, default=0.6
        PER prioritization exponent.
    per_beta : float, default=0.4
        Initial PER importance-sampling exponent.
    per_eps : float, default=1e-6
        Small additive constant used in PER priorities.
    per_beta_final : float, default=1.0
        Final PER importance-sampling exponent.
    per_beta_anneal_steps : int, default=200_000
        Annealing horizon from ``per_beta`` to ``per_beta_final``.

    Returns
    -------
    OffPolicyAlgorithm
        Fully assembled DDPG algorithm object.

    Raises
    ------
    ValueError
        If required dimensions or replay/update hyperparameters are invalid.
    """
    obs_dim_i = int(obs_dim)
    action_dim_i = int(action_dim)
    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
    buffer_size_i = int(buffer_size)
    batch_size_i = int(batch_size)
    update_after_i = int(update_after)
    update_every_i = int(update_every)
    gradient_steps_i = int(gradient_steps)
    max_updates_per_call_i = int(max_updates_per_call)

    if obs_dim_i <= 0:
        raise ValueError(f"obs_dim must be > 0, got {obs_dim_i}")
    if action_dim_i <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim_i}")
    if len(hidden_sizes_t) == 0 or any(h <= 0 for h in hidden_sizes_t):
        raise ValueError(f"hidden_sizes must be non-empty positive ints, got {hidden_sizes_t}")
    if buffer_size_i <= 0:
        raise ValueError(f"buffer_size must be > 0, got {buffer_size_i}")
    if batch_size_i <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size_i}")
    if update_every_i <= 0:
        raise ValueError(f"update_every must be > 0, got {update_every_i}")
    if gradient_steps_i < 0:
        raise ValueError(f"gradient_steps must be >= 0, got {gradient_steps_i}")

    noise = build_noise(
        kind=exploration_noise,
        action_dim=action_dim_i,
        device=device,
        noise_mu=float(noise_mu),
        noise_sigma=float(noise_sigma),
        ou_theta=float(ou_theta),
        ou_dt=float(ou_dt),
        uniform_low=float(uniform_low),
        uniform_high=float(uniform_high),
        action_noise_eps=float(action_noise_eps),
        action_noise_low=action_noise_low,
        action_noise_high=action_noise_high,
    )

    head = DDPGHead(
        obs_dim=obs_dim_i,
        action_dim=action_dim_i,
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
        action_low=action_low,
        action_high=action_high,
        noise=noise,
        noise_clip=None if noise_clip is None else float(noise_clip),
    )

    core = DDPGCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
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

    algo = OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=buffer_size_i,
        batch_size=batch_size_i,
        update_after=update_after_i,
        update_every=update_every_i,
        utd=float(utd),
        gradient_steps=gradient_steps_i,
        max_updates_per_call=max_updates_per_call_i,
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
        reset_noise_on_done=True,
    )
    return algo
