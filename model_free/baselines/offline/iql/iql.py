"""Factory entrypoint for offline Implicit Q-Learning (IQL).

The :func:`iql` function assembles:

- :class:`~rllib.model_free.baselines.offline.iql.head.IQLHead`
- :class:`~rllib.model_free.baselines.offline.iql.core.IQLCore`
- :class:`~rllib.model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`

The resulting object reuses common replay/update infrastructure while applying
IQL-specific value regression and advantage-weighted behavior cloning updates.
"""


from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.baselines.offline.iql.core import IQLCore
from rllib.model_free.baselines.offline.iql.head import IQLHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def iql(
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
    log_std_mode: str = "layer",
    log_std_init: float = -0.5,
    # ---------------------------------------------------------------------
    # IQL update (core)
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    expectile: float = 0.7,
    beta: float = 3.0,
    max_weight: float = 100.0,
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
    value_optim_name: str = "adamw",
    value_lr: float = 3e-4,
    value_weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) schedulers
    # ---------------------------------------------------------------------
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
    """Construct a configured offline IQL algorithm instance.

    Parameters
    ----------
    obs_dim : int
        Flattened observation feature dimension consumed by policy/value nets.
    action_dim : int
        Continuous action dimension.
    device : str or torch.device, default="cpu"
        Device used for model parameters and training updates.
    obs_shape : Any, optional
        Original observation shape for optional feature extractors.
    feature_extractor_cls : Any, optional
        Optional encoder/trunk module class used before MLP heads.
    feature_extractor_kwargs : Any, optional
        Keyword arguments passed to ``feature_extractor_cls``.
    init_trunk : Any, optional
        Optional trunk-initialization flag forwarded to network constructors.
    hidden_sizes : tuple of int, default=(256, 256)
        Hidden dimensions for actor, critic, and value networks.
    activation_fn : Any, default=torch.nn.ReLU
        Activation used throughout MLP blocks.
    init_type : str, default="orthogonal"
        Parameter initialization strategy.
    gain : float, default=1.0
        Gain for supported initialization paths.
    bias : float, default=0.0
        Initial bias value for supported layers.
    log_std_mode : str, default="layer"
        Actor log-standard-deviation parameterization mode.
    log_std_init : float, default=-0.5
        Initial actor log-standard-deviation value.
    gamma : float, default=0.99
        Discount factor for critic targets.
    tau : float, default=0.005
        Polyak averaging coefficient for target critic updates.
    target_update_interval : int, default=1
        Number of update steps between target updates.
    expectile : float, default=0.7
        Expectile used for value regression in IQL.
    beta : float, default=3.0
        Advantage temperature in actor's exponential weighting.
    max_weight : float, default=100.0
        Upper bound on exponentiated advantage weights.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold; ``0`` disables clipping.
    use_amp : bool, default=False
        Mixed-precision update toggle where supported.
    actor_optim_name, critic_optim_name, value_optim_name : str
        Optimizer names for actor, critic, and value modules.
    actor_lr, critic_lr, value_lr : float
        Learning rates for actor, critic, and value optimizers.
    actor_weight_decay, critic_weight_decay, value_weight_decay : float
        Weight decay values for each optimizer.
    actor_sched_name, critic_sched_name, value_sched_name : str
        Scheduler names (``"none"`` disables scheduling).
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Shared scheduler hyperparameters.
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Batch size for each gradient update.
    update_after : int, default=0
        Number of environment steps before training starts.
    update_every : int, default=1
        Environment-step interval between update calls.
    utd : float, default=1.0
        Update-to-data ratio used by the common off-policy wrapper.
    gradient_steps : int, default=1
        Number of gradient updates per update event.
    max_updates_per_call : int, default=1000
        Safety limit on updates done in a single training call.
    use_per : bool, default=True
        Whether prioritized replay integration is enabled.
    per_alpha : float, default=0.6
        Priority exponent for replay sampling.
    per_beta : float, default=0.4
        Initial importance-sampling exponent.
    per_eps : float, default=1e-6
        Epsilon added to priorities for numerical stability.
    per_beta_final : float, default=1.0
        Final annealed ``beta`` value.
    per_beta_anneal_steps : int, default=200000
        Annealing horizon for ``beta``.

    Returns
    -------
    OffPolicyAlgorithm
        Configured algorithm wrapper holding an IQL head/core pair.

    Raises
    ------
    ValueError
        If mandatory shape or scheduling parameters are invalid.
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

    head = IQLHead(
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
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
    )

    core = IQLCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        expectile=float(expectile),
        beta=float(beta),
        max_weight=float(max_weight),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        value_optim_name=str(value_optim_name),
        value_lr=float(value_lr),
        value_weight_decay=float(value_weight_decay),
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        value_sched_name=str(value_sched_name),
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
