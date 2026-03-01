"""Factory entrypoint for the offline Conservative Q-Learning algorithm.

The :func:`cql` function wires together:

- :class:`~rllib.model_free.baselines.offline.cql.head.CQLHead`
- :class:`~rllib.model_free.baselines.offline.cql.core.CQLCore`
- :class:`~rllib.model_free.common.policies.off_policy_algorithm.OffPolicyAlgorithm`

This mirrors the composition pattern used by other baselines in this repository
while exposing CQL-specific knobs (conservative penalty and optional adaptive
penalty coefficient).
"""


from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.baselines.offline.cql.core import CQLCore
from rllib.model_free.baselines.offline.cql.head import CQLHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm


def cql(
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
    # CQL / SAC update (core)
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    tau: float = 0.005,
    target_update_interval: int = 1,
    auto_alpha: bool = True,
    alpha_init: float = 0.2,
    target_entropy: Optional[float] = None,
    cql_n_actions: int = 10,
    cql_temp: float = 1.0,
    cql_alpha: float = 1.0,
    cql_target_action_gap: float = 0.0,
    auto_cql_alpha: bool = False,
    cql_alpha_init: float = 1.0,
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
    alpha_optim_name: str = "adamw",
    alpha_lr: float = 3e-4,
    alpha_weight_decay: float = 0.0,
    cql_alpha_optim_name: str = "adamw",
    cql_alpha_lr: float = 3e-4,
    cql_alpha_weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) schedulers
    # ---------------------------------------------------------------------
    actor_sched_name: str = "none",
    critic_sched_name: str = "none",
    alpha_sched_name: str = "none",
    cql_alpha_sched_name: str = "none",
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
    """Construct a configured offline CQL algorithm instance.

    Parameters
    ----------
    obs_dim : int
        Flattened observation feature dimension consumed by the head.
    action_dim : int
        Continuous action dimension.
    device : str or torch.device, default="cpu"
        Device where module parameters and update computation are placed.
    obs_shape : Any, optional
        Original observation shape for optional feature extractors.
    feature_extractor_cls : Any, optional
        Optional trunk/encoder class applied before MLP heads.
    feature_extractor_kwargs : Any, optional
        Keyword arguments forwarded to ``feature_extractor_cls``.
    init_trunk : Any, optional
        Optional initialization override for extractor/trunk parameters.
    hidden_sizes : tuple of int, default=(256, 256)
        Hidden layer sizes used for actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation module/callable used in network blocks.
    init_type : str, default="orthogonal"
        Parameter initialization strategy.
    gain : float, default=1.0
        Initialization gain applied to supported layers.
    bias : float, default=0.0
        Constant bias initialization value.
    log_std_mode : str, default="layer"
        Log-standard-deviation parameterization mode for the actor.
    log_std_init : float, default=-0.5
        Initial log standard deviation value.
    gamma : float, default=0.99
        Discount factor for Bellman targets.
    tau : float, default=0.005
        Polyak averaging factor for critic target updates.
    target_update_interval : int, default=1
        Frequency (in update steps) for target critic updates.
    auto_alpha : bool, default=True
        If ``True``, tune SAC entropy temperature automatically.
    alpha_init : float, default=0.2
        Initial entropy temperature value.
    target_entropy : float, optional
        Target policy entropy. If ``None``, defaults to ``-action_dim``.
    cql_n_actions : int, default=10
        Number of sampled out-of-distribution actions per state for CQL penalty.
    cql_temp : float, default=1.0
        Temperature in conservative log-sum-exp aggregation.
    cql_alpha : float, default=1.0
        Fixed CQL penalty multiplier when ``auto_cql_alpha=False``.
    cql_target_action_gap : float, default=0.0
        Target conservative gap used for adaptive CQL alpha updates.
    auto_cql_alpha : bool, default=False
        If ``True``, learn the CQL penalty multiplier.
    cql_alpha_init : float, default=1.0
        Initial value for learned CQL penalty multiplier.
    max_grad_norm : float, default=0.0
        Global gradient clipping threshold; ``0`` disables clipping.
    use_amp : bool, default=False
        Whether to enable mixed-precision update flow where supported.
    actor_optim_name, critic_optim_name, alpha_optim_name, cql_alpha_optim_name : str
        Optimizer names for each trainable component.
    actor_lr, critic_lr, alpha_lr, cql_alpha_lr : float
        Learning rates for each optimizer.
    actor_weight_decay, critic_weight_decay, alpha_weight_decay, cql_alpha_weight_decay : float
        Weight decay values for each optimizer.
    actor_sched_name, critic_sched_name, alpha_sched_name, cql_alpha_sched_name : str
        Scheduler names for each optimizer (``"none"`` disables scheduling).
    total_steps, warmup_steps, min_lr_ratio, poly_power, step_size, sched_gamma, milestones
        Scheduler configuration shared across optimizer schedulers.
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    batch_size : int, default=256
        Minibatch size sampled per gradient update.
    update_after : int, default=0
        Number of environment steps before updates start.
    update_every : int, default=1
        Update frequency in environment steps.
    utd : float, default=1.0
        Update-to-data ratio used by the off-policy wrapper.
    gradient_steps : int, default=1
        Number of gradient updates per update event.
    max_updates_per_call : int, default=1000
        Safety cap on updates performed in one training call.
    use_per : bool, default=True
        Enable prioritized replay integration.
    per_alpha : float, default=0.6
        Prioritization exponent for replay sampling.
    per_beta : float, default=0.4
        Initial importance-sampling correction exponent.
    per_eps : float, default=1e-6
        Small epsilon added to TD errors for stable priorities.
    per_beta_final : float, default=1.0
        Final beta value after annealing.
    per_beta_anneal_steps : int, default=200000
        Number of steps used to anneal PER beta.

    Returns
    -------
    OffPolicyAlgorithm
        Configured algorithm object combining CQL head/core with the common
        off-policy training loop and replay scheduling.

    Raises
    ------
    ValueError
        If critical shape or scheduling parameters are invalid.
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

    head = CQLHead(
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

    core = CQLCore(
        head=head,
        gamma=float(gamma),
        tau=float(tau),
        target_update_interval=int(target_update_interval),
        auto_alpha=bool(auto_alpha),
        alpha_init=float(alpha_init),
        target_entropy=(None if target_entropy is None else float(target_entropy)),
        cql_n_actions=int(cql_n_actions),
        cql_temp=float(cql_temp),
        cql_alpha=float(cql_alpha),
        cql_target_action_gap=float(cql_target_action_gap),
        auto_cql_alpha=bool(auto_cql_alpha),
        cql_alpha_init=float(cql_alpha_init),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        actor_optim_name=str(actor_optim_name),
        actor_lr=float(actor_lr),
        actor_weight_decay=float(actor_weight_decay),
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        alpha_optim_name=str(alpha_optim_name),
        alpha_lr=float(alpha_lr),
        alpha_weight_decay=float(alpha_weight_decay),
        cql_alpha_optim_name=str(cql_alpha_optim_name),
        cql_alpha_lr=float(cql_alpha_lr),
        cql_alpha_weight_decay=float(cql_alpha_weight_decay),
        actor_sched_name=str(actor_sched_name),
        critic_sched_name=str(critic_sched_name),
        alpha_sched_name=str(alpha_sched_name),
        cql_alpha_sched_name=str(cql_alpha_sched_name),
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
