"""C51 Builder.

This module provides a high-level constructor for the C51 (Categorical DQN)
baseline in discrete action spaces.

The builder composes:

1. :class:`C51Head` for online/target distributional Q-networks.
2. :class:`C51Core` for categorical Bellman projection and optimization.
3. :class:`OffPolicyAlgorithm` for replay-driven training orchestration.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import torch as th

from rllib.model_free.common.utils.common_utils import _to_pos_int
import torch.nn as nn

from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm
from rllib.model_free.baselines.value_based.c51.core import C51Core
from rllib.model_free.baselines.value_based.c51.head import C51Head



def c51(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    atom_size: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    activation_fn: Any = nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    noisy_std_init: float = 0.0,
    # ---------------------------------------------------------------------
    # C51 update (core) hyperparameters
    # ---------------------------------------------------------------------
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    tau: float = 0.0,
    double_dqn: bool = True,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    # ---------------------------------------------------------------------
    # (Optional) scheduler
    # ---------------------------------------------------------------------
    sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # ---------------------------------------------------------------------
    # OffPolicyAlgorithm schedule / replay
    # ---------------------------------------------------------------------
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    update_after: int = 1_000,
    update_every: int = 1,
    utd: float = 1.0,
    gradient_steps: int = 1,
    max_updates_per_call: int = 1_000,
    # ---------------------------------------------------------------------
    # PER (Prioritized Experience Replay)
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
    # ---------------------------------------------------------------------
    # Exploration (epsilon-greedy)
    # ---------------------------------------------------------------------
    exploration_eps: float = 0.1,
    exploration_eps_final: float = 0.05,
    exploration_eps_anneal_steps: int = 200_000,
    exploration_eval_eps: float = 0.0,
) -> OffPolicyAlgorithm:
    """Build a complete C51 algorithm (head + core + off-policy driver).

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    n_actions : int
        Number of discrete actions.
    device : str | torch.device, default="cpu"
        Learner device.
    atom_size : int, default=51
        Number of categorical support atoms.
    v_min, v_max : float
        Lower/upper support bounds.
    hidden_sizes : tuple[int, ...], default=(256, 256)
        MLP hidden widths for distributional Q-networks.
    gamma : float, default=0.99
        Discount factor.
    target_update_interval : int, default=1000
        Hard target update interval when ``tau=0``.
    tau : float, default=0.0
        Polyak coefficient for soft updates when > 0.
    double_dqn : bool, default=True
        Whether to use Double-DQN action selection for next-state distribution.
    use_per : bool, default=True
        Enable prioritized replay.
    use_her : bool, default=False
        Enable HER replay wrapper.

    Returns
    -------
    OffPolicyAlgorithm
        Fully configured off-policy training driver for C51.
    """
    obs_dim_i = _to_pos_int("obs_dim", obs_dim)
    n_actions_i = _to_pos_int("n_actions", n_actions)
    atom_size_i = int(atom_size)
    if atom_size_i < 2:
        raise ValueError(f"atom_size must be >= 2, got: {atom_size}")
    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
    if not float(v_min) < float(v_max):
        raise ValueError(f"Require v_min < v_max, got v_min={v_min}, v_max={v_max}")
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
        raise ValueError(f"per_beta_anneal_steps must be >= 0, got: {per_beta_anneal_steps}")
    exploration_eps_anneal_steps_i = int(exploration_eps_anneal_steps)
    if exploration_eps_anneal_steps_i < 0:
        raise ValueError(
            f"exploration_eps_anneal_steps must be >= 0, got: {exploration_eps_anneal_steps}"
        )
    milestones_t = tuple(int(m) for m in milestones)

    head = C51Head(
        obs_dim=obs_dim_i,
        n_actions=n_actions_i,
        atom_size=atom_size_i,
        v_min=float(v_min),
        v_max=float(v_max),
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        noisy_std_init=float(noisy_std_init),
        device=device,
    )

    core = C51Core(
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

    algo = OffPolicyAlgorithm(
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
    return algo
