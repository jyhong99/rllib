"""FQF builder for discrete-action off-policy training.

This module exposes :func:`fqf`, a convenience constructor that wires together
three components:

- :class:`FQFHead` for quantile and fraction networks.
- :class:`FQFCore` for Fractional Quantile Function optimization.
- :class:`OffPolicyAlgorithm` for replay, exploration, and update scheduling.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th

from rllib.model_free.common.utils.common_utils import _to_pos_int
import torch.nn as nn

from rllib.model_free.baselines.value_based.fqf.core import FQFCore
from rllib.model_free.baselines.value_based.fqf.head import FQFHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm



def fqf(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # network
    n_quantiles: int = 32,
    n_cos_embeddings: int = 64,
    fraction_hidden_size: int = 128,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # core
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    tau: float = 0.0,
    double_dqn: bool = True,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    fraction_entropy_coef: float = 0.0,
    # quantile optimizer
    optim_name: str = "adamw",
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    # fraction optimizer
    fraction_optim_name: str = "adamw",
    fraction_lr: float = 2.5e-4,
    fraction_weight_decay: float = 0.0,
    # schedulers
    sched_name: str = "none",
    fraction_sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # off-policy
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
    # HER
    use_her: bool = False,
    her_goal_shape: Any = None,
    her_reward_fn: Any = None,
    her_done_fn: Any = None,
    her_ratio: float = 0.8,
    # exploration
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
) -> OffPolicyAlgorithm:
    """Build an FQF algorithm instance for discrete control.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension.
    n_actions : int
        Number of discrete actions.
    device : str | torch.device, default="cpu"
        Device used by head, core, and algorithm wrappers.
    obs_shape : Any, default=None
        Optional original observation shape used when building feature
        extractors.
    feature_extractor_cls : Any, default=None
        Optional feature extractor class used by quantile networks.
    feature_extractor_kwargs : Any, default=None
        Keyword arguments for ``feature_extractor_cls``.
    init_trunk : Any, default=None
        Optional flag controlling whether external feature extractor modules are
        re-initialized.
    n_quantiles : int, default=32
        Number of quantile fractions optimized by FQF.
    n_cos_embeddings : int, default=64
        Number of cosine embeddings for tau conditioning.
    fraction_hidden_size : int, default=128
        Hidden width of the fraction proposal network.
    hidden_sizes : tuple[int, ...], default=(256, 256)
        Shared MLP hidden sizes for quantile value heads.
    activation_fn : Any, default=torch.nn.ReLU
        Activation module class used across networks.
    dueling_mode : bool, default=False
        Whether to use dueling action-value decomposition.
    init_type : str, default="orthogonal"
        Parameter initializer identifier.
    gain : float, default=1.0
        Initialization gain.
    bias : float, default=0.0
        Initialization bias constant.
    gamma : float, default=0.99
        Discount factor.
    target_update_interval : int, default=1000
        Hard target update period when ``tau == 0``.
    tau : float, default=0.0
        Soft update factor in ``[0, 1]``.
    double_dqn : bool, default=True
        Use online argmax with target quantiles for bootstrap action selection.
    max_grad_norm : float, default=0.0
        Global gradient clipping threshold (0 disables clipping).
    use_amp : bool, default=False
        Enable mixed precision for quantile network optimization.
    fraction_entropy_coef : float, default=0.0
        Coefficient for optional entropy regularization in fraction loss.
    optim_name : str, default="adamw"
        Quantile-network optimizer name.
    lr : float, default=5e-5
        Quantile-network learning rate.
    weight_decay : float, default=0.0
        Quantile-network optimizer weight decay.
    fraction_optim_name : str, default="adamw"
        Fraction-network optimizer name.
    fraction_lr : float, default=2.5e-4
        Fraction-network learning rate.
    fraction_weight_decay : float, default=0.0
        Fraction-network weight decay.
    sched_name : str, default="none"
        Scheduler name for quantile optimizer.
    fraction_sched_name : str, default="none"
        Scheduler name for fraction optimizer.
    total_steps : int, default=0
        Scheduler horizon.
    warmup_steps : int, default=0
        Scheduler warmup length.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for schedulers that support floor ratios.
    poly_power : float, default=1.0
        Polynomial scheduler exponent.
    step_size : int, default=1000
        Step scheduler period.
    sched_gamma : float, default=0.99
        Multiplicative scheduler decay.
    milestones : Sequence[int], default=()
        Multi-step scheduler milestone indices.
    buffer_size : int, default=1_000_000
        Replay capacity.
    batch_size : int, default=256
        Minibatch size per update.
    update_after : int, default=1_000
        Number of transitions collected before updates begin.
    update_every : int, default=1
        Environment steps between update calls.
    utd : float, default=1.0
        Update-to-data ratio when `gradient_steps <= 0`.
    gradient_steps : int, default=1
        Explicit number of gradient updates per train call.
    max_updates_per_call : int, default=1_000
        Safety cap on updates per train call.
    use_per : bool, default=True
        Enable prioritized replay.
    per_alpha : float, default=0.6
        Prioritization exponent.
    per_beta : float, default=0.4
        Initial PER importance-sampling exponent.
    per_eps : float, default=1e-6
        Numerical epsilon used in PER and TD error clamping.
    per_beta_final : float, default=1.0
        Final PER beta value.
    per_beta_anneal_steps : int, default=200_000
        Annealing horizon for beta.
    use_her : bool, default=False
        Enable hindsight experience replay.
    her_goal_shape : Any, default=None
        Goal tensor shape for HER relabeling.
    her_reward_fn : Any, default=None
        Reward recomputation callable used by HER.
    her_done_fn : Any, default=None
        Done recomputation callable used by HER.
    her_ratio : float, default=0.8
        Fraction of HER-relabeled samples.
    exploration_eps : float, default=0.1
        Initial epsilon for epsilon-greedy data collection.
    exploration_eps_final : float, default=0.05
        Final epsilon after annealing.
    exploration_eps_anneal_steps : int, default=200_000
        Annealing horizon for epsilon.
    exploration_eval_eps : float, default=0.0
        Epsilon used during evaluation rollouts.

    Returns
    -------
    OffPolicyAlgorithm
        Configured FQF training algorithm wrapper.

    Raises
    ------
    ValueError
        If integer dimensions or schedule parameters are invalid.
    """
    obs_dim_i = _to_pos_int("obs_dim", obs_dim)
    n_actions_i = _to_pos_int("n_actions", n_actions)
    n_quantiles_i = _to_pos_int("n_quantiles", n_quantiles)
    if n_quantiles_i <= 1:
        raise ValueError(f"n_quantiles must be > 1 for FQF, got: {n_quantiles_i}")
    n_cos_embeddings_i = _to_pos_int("n_cos_embeddings", n_cos_embeddings)
    fraction_hidden_size_i = _to_pos_int("fraction_hidden_size", fraction_hidden_size)

    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
    if not hidden_sizes_t or any(h <= 0 for h in hidden_sizes_t):
        raise ValueError(f"hidden_sizes must contain positive integers, got: {hidden_sizes_t}")

    buffer_size_i = _to_pos_int("buffer_size", buffer_size)
    batch_size_i = _to_pos_int("batch_size", batch_size)
    update_every_i = _to_pos_int("update_every", update_every)

    gradient_steps_i = int(gradient_steps)
    if gradient_steps_i < 0:
        raise ValueError(f"gradient_steps must be >= 0, got: {gradient_steps_i}")

    per_beta_anneal_steps_i = int(per_beta_anneal_steps)
    if per_beta_anneal_steps_i < 0:
        raise ValueError(f"per_beta_anneal_steps must be >= 0, got: {per_beta_anneal_steps_i}")

    exploration_eps_anneal_steps_i = int(exploration_eps_anneal_steps)
    if exploration_eps_anneal_steps_i < 0:
        raise ValueError(
            f"exploration_eps_anneal_steps must be >= 0, got: {exploration_eps_anneal_steps_i}"
        )

    milestones_t = tuple(int(m) for m in milestones)

    head = FQFHead(
        obs_dim=obs_dim_i,
        n_actions=n_actions_i,
        n_quantiles=n_quantiles_i,
        n_cos_embeddings=n_cos_embeddings_i,
        fraction_hidden_size=fraction_hidden_size_i,
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    core = FQFCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        per_eps=float(per_eps),
        fraction_entropy_coef=float(fraction_entropy_coef),
        optim_name=str(optim_name),
        lr=float(lr),
        weight_decay=float(weight_decay),
        fraction_optim_name=str(fraction_optim_name),
        fraction_lr=float(fraction_lr),
        fraction_weight_decay=float(fraction_weight_decay),
        sched_name=str(sched_name),
        fraction_sched_name=str(fraction_sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=milestones_t,
    )

    return OffPolicyAlgorithm(
        head=head,
        core=core,
        device=device,
        buffer_size=buffer_size_i,
        batch_size=batch_size_i,
        update_after=int(update_after),
        update_every=update_every_i,
        utd=float(utd),
        gradient_steps=gradient_steps_i,
        max_updates_per_call=int(max_updates_per_call),
        use_per=bool(use_per),
        per_alpha=float(per_alpha),
        per_beta=float(per_beta),
        per_eps=float(per_eps),
        per_beta_final=float(per_beta_final),
        per_beta_anneal_steps=per_beta_anneal_steps_i,
        use_her=bool(use_her),
        her_goal_shape=her_goal_shape,
        her_reward_fn=her_reward_fn,
        her_done_fn=her_done_fn,
        her_ratio=float(her_ratio),
        exploration_eps=float(exploration_eps),
        exploration_eps_final=float(exploration_eps_final),
        exploration_eps_anneal_steps=exploration_eps_anneal_steps_i,
        exploration_eval_eps=float(exploration_eval_eps),
    )
