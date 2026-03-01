"""QR-DQN builder for discrete-action off-policy training.

This module exposes :func:`qrdqn`, a convenience constructor that wires:

- :class:`QRDQNHead` for fixed-quantile online/target Q-networks.
- :class:`QRDQNCore` for quantile regression TD optimization.
- :class:`OffPolicyAlgorithm` for replay and update scheduling.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th

from rllib.model_free.common.utils.common_utils import _to_pos_int
import torch.nn as nn

from rllib.model_free.baselines.value_based.qrdqn.core import QRDQNCore
from rllib.model_free.baselines.value_based.qrdqn.head import QRDQNHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm



def qrdqn(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # network
    n_quantiles: int = 200,
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
    # optimizer
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    # scheduler
    sched_name: str = "none",
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
    """Build a QR-DQN algorithm wrapper.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension.
    n_actions : int
        Number of discrete actions.
    device : str | torch.device, default="cpu"
        Device used by head, core, and algorithm wrapper.
    obs_shape : Any, default=None
        Optional original observation shape for feature extractor construction.
    feature_extractor_cls : Any, default=None
        Optional feature extractor class for quantile networks.
    feature_extractor_kwargs : Any, default=None
        Optional keyword arguments for ``feature_extractor_cls``.
    init_trunk : Any, default=None
        Optional external trunk initialization control.
    n_quantiles : int, default=200
        Number of fixed quantiles per action.
    hidden_sizes : tuple[int, ...], default=(256, 256)
        Hidden MLP widths for quantile value networks.
    activation_fn : Any, default=torch.nn.ReLU
        Activation module class.
    dueling_mode : bool, default=False
        Enable dueling decomposition in quantile networks.
    init_type : str, default="orthogonal"
        Parameter initializer name.
    gain : float, default=1.0
        Initializer gain.
    bias : float, default=0.0
        Initializer bias constant.
    gamma : float, default=0.99
        Discount factor.
    target_update_interval : int, default=1000
        Hard target update period when ``tau == 0``.
    tau : float, default=0.0
        Polyak averaging factor in ``[0, 1]``.
    double_dqn : bool, default=True
        Whether to use Double-DQN action selection.
    max_grad_norm : float, default=0.0
        Gradient clipping threshold; non-positive disables clipping.
    use_amp : bool, default=False
        Enable mixed precision training.
    optim_name : str, default="adamw"
        Optimizer name.
    lr : float, default=3e-4
        Learning rate.
    weight_decay : float, default=0.0
        Weight decay.
    sched_name : str, default="none"
        Scheduler type.
    total_steps : int, default=0
        Scheduler horizon.
    warmup_steps : int, default=0
        Scheduler warmup steps.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio for supported schedulers.
    poly_power : float, default=1.0
        Polynomial scheduler exponent.
    step_size : int, default=1000
        Step scheduler interval.
    sched_gamma : float, default=0.99
        Multiplicative scheduler factor.
    milestones : Sequence[int], default=()
        Milestones for multi-step scheduler.
    buffer_size : int, default=1_000_000
        Replay capacity.
    batch_size : int, default=256
        Batch size per gradient step.
    update_after : int, default=1_000
        Number of collected transitions before updates begin.
    update_every : int, default=1
        Environment steps between update calls.
    utd : float, default=1.0
        Update-to-data ratio when ``gradient_steps <= 0``.
    gradient_steps : int, default=1
        Explicit gradient update count per call.
    max_updates_per_call : int, default=1_000
        Safety cap on updates per call.
    use_per : bool, default=True
        Enable prioritized replay.
    per_alpha : float, default=0.6
        PER prioritization exponent.
    per_beta : float, default=0.4
        Initial PER importance correction exponent.
    per_eps : float, default=1e-6
        Numerical epsilon for PER and TD-error clamping.
    per_beta_final : float, default=1.0
        Final PER beta value.
    per_beta_anneal_steps : int, default=200_000
        Annealing horizon for beta.
    use_her : bool, default=False
        Enable hindsight experience replay.
    her_goal_shape : Any, default=None
        Goal tensor shape for HER relabeling.
    her_reward_fn : Any, default=None
        Reward recomputation callback for HER.
    her_done_fn : Any, default=None
        Done recomputation callback for HER.
    her_ratio : float, default=0.8
        Fraction of HER relabeled samples.
    exploration_eps : float, default=0.1
        Initial epsilon for epsilon-greedy exploration.
    exploration_eps_final : float, default=0.05
        Final epsilon after annealing.
    exploration_eps_anneal_steps : int, default=200_000
        Epsilon annealing horizon.
    exploration_eval_eps : float, default=0.0
        Evaluation-time epsilon.

    Returns
    -------
    OffPolicyAlgorithm
        Configured QR-DQN algorithm wrapper.

    Raises
    ------
    ValueError
        If integer hyperparameters are invalid.
    """
    obs_dim_i = _to_pos_int("obs_dim", obs_dim)
    n_actions_i = _to_pos_int("n_actions", n_actions)
    n_quantiles_i = _to_pos_int("n_quantiles", n_quantiles)

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

    head = QRDQNHead(
        obs_dim=obs_dim_i,
        n_actions=n_actions_i,
        n_quantiles=n_quantiles_i,
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        dueling_mode=bool(dueling_mode),
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    core = QRDQNCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        optim_name=str(optim_name),
        lr=float(lr),
        weight_decay=float(weight_decay),
        sched_name=str(sched_name),
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=milestones_t,
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        per_eps=float(per_eps),
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
