"""DRQN Builder.

This module provides a high-level constructor for DRQN (recurrent DQN) in
discrete action spaces.

The builder composes:

1. :class:`DRQNHead` with online/target recurrent Q-networks.
2. :class:`DRQNCore` with sequence-aware TD updates.
3. :class:`OffPolicyAlgorithm` for replay-driven training orchestration.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

import torch as th

from rllib.model_free.common.utils.common_utils import _to_pos_int

from rllib.model_free.baselines.value_based.drqn.core import DRQNCore
from rllib.model_free.baselines.value_based.drqn.head import DRQNHead
from rllib.model_free.common.policies.off_policy_algorithm import OffPolicyAlgorithm



def drqn(
    *,
    obs_dim: int,
    n_actions: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    # network
    hidden_sizes: Tuple[int, ...] = (128,),
    rnn_hidden_size: int = 128,
    rnn_num_layers: int = 1,
    activation_fn: Any = th.nn.ReLU,
    dueling_mode: bool = False,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    # core
    gamma: float = 0.99,
    double_dqn: bool = True,
    huber: bool = True,
    target_update_interval: int = 1000,
    tau: float = 0.0,
    max_grad_norm: float = 0.0,
    use_amp: bool = False,
    per_eps: float = 1e-6,
    # optimizer/scheduler
    optim_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    sched_name: str = "none",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Tuple[int, ...] = (),
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
    per_beta_final: float = 1.0,
    per_beta_anneal_steps: int = 200_000,
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
    # sequence replay
    replay_sequence_length: int = 8,
    replay_sequence_strict_done: bool = True,
) -> OffPolicyAlgorithm:
    """Build DRQN as ``Head + Core + OffPolicyAlgorithm``.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    n_actions : int
        Number of discrete actions.
    rnn_hidden_size : int, default=128
        Hidden size of recurrent Q-network state.
    rnn_num_layers : int, default=1
        Number of recurrent layers.
    replay_sequence_length : int, default=8
        Temporal unroll length sampled from replay.
    replay_sequence_strict_done : bool, default=True
        Whether sequence extraction is constrained by episode boundaries.

    Returns
    -------
    OffPolicyAlgorithm
        Fully configured off-policy training driver for DRQN.
    """
    obs_dim_i = _to_pos_int("obs_dim", obs_dim)
    n_actions_i = _to_pos_int("n_actions", n_actions)
    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
    if not hidden_sizes_t or any(h <= 0 for h in hidden_sizes_t):
        raise ValueError(f"hidden_sizes must contain positive integers, got: {hidden_sizes_t}")
    rnn_hidden_size_i = _to_pos_int("rnn_hidden_size", rnn_hidden_size)
    rnn_num_layers_i = _to_pos_int("rnn_num_layers", rnn_num_layers)
    buffer_size_i = _to_pos_int("buffer_size", buffer_size)
    batch_size_i = _to_pos_int("batch_size", batch_size)
    update_every_i = _to_pos_int("update_every", update_every)
    gradient_steps_i = int(gradient_steps)
    if gradient_steps_i < 0:
        raise ValueError(f"gradient_steps must be >= 0, got: {gradient_steps_i}")
    replay_sequence_length_i = _to_pos_int("replay_sequence_length", replay_sequence_length)
    milestones_t = tuple(int(m) for m in milestones)
    per_beta_anneal_steps_i = int(per_beta_anneal_steps)
    if per_beta_anneal_steps_i < 0:
        raise ValueError(f"per_beta_anneal_steps must be >= 0, got: {per_beta_anneal_steps}")
    exploration_eps_anneal_steps_i = int(exploration_eps_anneal_steps)
    if exploration_eps_anneal_steps_i < 0:
        raise ValueError(
            f"exploration_eps_anneal_steps must be >= 0, got: {exploration_eps_anneal_steps}"
        )

    head = DRQNHead(
        obs_dim=obs_dim_i,
        n_actions=n_actions_i,
        hidden_sizes=hidden_sizes_t,
        rnn_hidden_size=rnn_hidden_size_i,
        rnn_num_layers=rnn_num_layers_i,
        activation_fn=activation_fn,
        dueling_mode=bool(dueling_mode),
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        device=device,
    )

    core = DRQNCore(
        head=head,
        gamma=float(gamma),
        target_update_interval=int(target_update_interval),
        tau=float(tau),
        double_dqn=bool(double_dqn),
        huber=bool(huber),
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        per_eps=float(per_eps),
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
        replay_sequence_length=replay_sequence_length_i,
        replay_sequence_strict_done=bool(replay_sequence_strict_done),
    )
