"""TRPO Builder.

This module provides a high-level constructor for the continuous-action TRPO
baseline.

The builder composes:

1. :class:`TRPOHead` for actor/critic networks.
2. :class:`TRPOCore` for trust-region policy updates and critic regression.
3. :class:`OnPolicyAlgorithm` for rollout collection and update scheduling.

Notes
-----
TRPO-specific natural-gradient logic (CG solve, FVP, KL-constrained line
search) is implemented in :class:`TRPOCore`.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union, Optional

import numpy as np
import torch as th

from rllib.model_free.common.utils.common_utils import _to_pos_int

from rllib.model_free.baselines.policy_based.on_policy.trpo.head import TRPOHead
from rllib.model_free.baselines.policy_based.on_policy.trpo.core import TRPOCore
from rllib.model_free.common.policies.on_policy_algorithm import OnPolicyAlgorithm



def trpo(
    *,
    obs_dim: int,
    action_dim: int,
    device: Union[str, th.device] = "cpu",
    obs_shape: Any = None,
    feature_extractor_cls: Any = None,
    feature_extractor_kwargs: Any = None,
    init_trunk: Any = None,
    # ---------------------------------------------------------------------
    # Network (head) hyperparameters
    # ---------------------------------------------------------------------
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation_fn: Any = th.nn.ReLU,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias: float = 0.0,
    log_std_mode: str = "param",
    log_std_init: float = -0.5,
    # ---------------------------------------------------------------------
    # TRPO update (core) hyperparameters
    # ---------------------------------------------------------------------
    max_kl: float = 1e-2,
    cg_iters: int = 10,
    cg_damping: float = 1e-2,
    backtrack_iters: int = 10,
    backtrack_coeff: float = 0.8,
    accept_ratio: float = 0.1,
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    # ---------------------------------------------------------------------
    # Critic optimizer / scheduler
    # ---------------------------------------------------------------------
    critic_optim_name: str = "adamw",
    critic_lr: float = 3e-4,
    critic_weight_decay: float = 0.0,
    critic_sched_name: str = "none",
    # Scheduler knobs (shared across cores/builders; optional)
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    poly_power: float = 1.0,
    step_size: int = 1000,
    sched_gamma: float = 0.99,
    milestones: Sequence[int] = (),
    # ---------------------------------------------------------------------
    # OnPolicyAlgorithm rollout / schedule
    # ---------------------------------------------------------------------
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 1,
    minibatch_size: Optional[int] = None,
    dtype_obs: Any = np.float32,
    dtype_act: Any = np.float32,
    normalize_advantages: bool = False,
    adv_eps: float = 1e-8,
) -> OnPolicyAlgorithm:
    """
    Build a complete TRPO `OnPolicyAlgorithm` instance (builder function).

    This function wires together three components into a ready-to-use algorithm:

    1) **TRPOHead**
       - Contains the actor (Gaussian policy) and critic (state-value function).
       - Owns device placement and model initialization.

    2) **TRPOCore**
       - Implements the TRPO update rule:
         * critic regression on returns (standard supervised loss),
         * natural gradient for the actor via conjugate gradient (CG),
         * KL trust-region enforcement using backtracking line search.

    3) **OnPolicyAlgorithm**
       - Collects rollouts from the environment,
       - Computes returns and GAE advantages,
       - Calls `core.update_from_batch(...)` according to update scheduling.

    Notes
    -----
    - Classic TRPO is typically *full-batch*: one rollout batch, one policy update.
      Therefore `update_epochs=1` and `minibatch_size=None` are recommended.
    - Some training loops require an explicit integer minibatch size. If your
      `OnPolicyAlgorithm` does not accept `None`, set:
      `minibatch_size=rollout_steps` to emulate full-batch behavior.
    - AMP (`use_amp`) is only applied to the critic regression path. The policy
      update requires higher-order derivatives (Fisher-vector products), which is
      usually not compatible with AMP/autocast in a stable way.

    Parameters
    ----------
    obs_dim : int
        Observation (state) dimension.
    action_dim : int
        Continuous action dimension.
    device : Union[str, torch.device], default="cpu"
        Torch device used by head/core/algo (e.g., "cpu", "cuda", torch.device("cuda")).

    hidden_sizes : Tuple[int, ...], default=(64, 64)
        Hidden layer sizes for both actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class used in MLPs (e.g., torch.nn.Tanh, torch.nn.ReLU).
        This is typically a **class**, not an instantiated module.
    init_type : str, default="orthogonal"
        Weight initialization strategy understood by your `TRPOHead` implementation.
    gain : float, default=1.0
        Initialization gain multiplier (passed to your init routine).
    bias : float, default=0.0
        Bias initialization value (passed to your init routine).
    log_std_mode : str, default="param"
        Policy log-std parameterization mode for Gaussian policy. Common choices:
        - "param": global learnable parameter
        - "net": predicted by network
        (Exact semantics depend on your `TRPOHead`.)
    log_std_init : float, default=-0.5
        Initial log standard deviation for the Gaussian policy.

    max_kl : float, default=1e-2
        Maximum allowed **mean** KL divergence per TRPO update (trust region size).
    cg_iters : int, default=10
        Number of conjugate gradient iterations for solving the natural gradient system.
    cg_damping : float, default=1e-2
        Damping added to the Fisher-vector product for numerical stability.
    backtrack_iters : int, default=10
        Maximum backtracking iterations in line search.
    backtrack_coeff : float, default=0.8
        Multiplicative step-size shrink factor during line search (must be in (0, 1)).
    accept_ratio : float, default=0.1
        Minimum ratio of (actual improvement / expected improvement) required to accept a step.
        Higher values make the line search stricter.
    max_grad_norm : float, default=0.5
        Gradient norm clipping threshold used for the critic update (and any clipped paths
        implemented by the core). Set to 0 to disable clipping.
    use_amp : bool, default=False
        Enable AMP for critic regression. Policy update still uses full precision.

    critic_optim_name : str, default="adamw"
        Optimizer name for critic (resolved by `build_optimizer`).
    critic_lr : float, default=3e-4
        Critic learning rate.
    critic_weight_decay : float, default=0.0
        Critic weight decay.
    critic_sched_name : str, default="none"
        Scheduler name for critic (resolved by `build_scheduler`).
    total_steps : int, default=0
        Total training steps for schedulers that need it (optional; depends on scheduler).
    warmup_steps : int, default=0
        Warmup steps for schedulers that support warmup.
    min_lr_ratio : float, default=0.0
        Minimum LR ratio (e.g., final_lr = base_lr * min_lr_ratio) for certain schedulers.
    poly_power : float, default=1.0
        Polynomial decay power for polynomial schedulers.
    step_size : int, default=1000
        Step size for step schedulers.
    sched_gamma : float, default=0.99
        Multiplicative gamma for step schedulers.
    milestones : Sequence[int], default=()
        Milestones for multi-step schedulers.

    rollout_steps : int, default=2048
        Number of environment steps collected per update (per rollout).
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE lambda for advantage estimation.
    update_epochs : int, default=1
        Number of epochs over the rollout batch per update. TRPO typically uses 1.
    minibatch_size : Optional[int], default=None
        Minibatch size for updates. TRPO is typically full-batch; keep `None` unless your
        `OnPolicyAlgorithm` requires an integer.
    dtype_obs : Any, default=np.float32
        Observation dtype used by rollout buffer / preprocessing.
    dtype_act : Any, default=np.float32
        Action dtype used by rollout buffer / preprocessing.
    normalize_advantages : bool, default=False
        Whether to normalize advantages before policy update (can improve stability).
    adv_eps : float, default=1e-8
        Epsilon used in advantage normalization: std + adv_eps.

    Returns
    -------
    algo : OnPolicyAlgorithm
        Fully constructed TRPO algorithm instance.

    Examples
    --------
    >>> algo = trpo(obs_dim=24, action_dim=4, device="cuda")
    >>> algo.setup(env)
    >>> obs = env.reset()
    >>> while True:
    ...     action = algo.act(obs)
    ...     next_obs, reward, done, info = env.step(action)
    ...     algo.on_env_step(obs=obs, action=action, reward=reward, done=done, info=info)
    ...     if algo.ready_to_update():
    ...         metrics = algo.update()
    ...     obs = next_obs
    """
    obs_dim = _to_pos_int("obs_dim", obs_dim)
    action_dim = _to_pos_int("action_dim", action_dim)
    hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
    if len(hidden_sizes_t) == 0 or any(h <= 0 for h in hidden_sizes_t):
        raise ValueError(f"hidden_sizes must be non-empty positive ints, got {hidden_sizes_t}")
    rollout_steps_i = _to_pos_int("rollout_steps", rollout_steps)
    update_epochs_i = _to_pos_int("update_epochs", update_epochs)
    minibatch_size_i = None if minibatch_size is None else _to_pos_int("minibatch_size", minibatch_size)
    milestones_t = tuple(int(m) for m in milestones)

    # ---------------------------------------------------------------------
    # 1) Head: actor (Gaussian policy) + critic (V(s))
    # ---------------------------------------------------------------------
    head = TRPOHead(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes_t,
        activation_fn=activation_fn,
        obs_shape=obs_shape,
        feature_extractor_cls=feature_extractor_cls,
        feature_extractor_kwargs=feature_extractor_kwargs,
        init_trunk=init_trunk,
        init_type=str(init_type),
        gain=float(gain),
        bias=float(bias),
        log_std_mode=str(log_std_mode),
        log_std_init=float(log_std_init),
        device=device,
    )

    # ---------------------------------------------------------------------
    # 2) Core: TRPO update engine (critic regression + natural gradient step)
    # ---------------------------------------------------------------------
    core = TRPOCore(
        head=head,
        # TRPO / natural gradient knobs
        max_kl=float(max_kl),
        cg_iters=int(cg_iters),
        cg_damping=float(cg_damping),
        # backtracking line search knobs
        backtrack_iters=int(backtrack_iters),
        backtrack_coeff=float(backtrack_coeff),
        accept_ratio=float(accept_ratio),
        # misc
        max_grad_norm=float(max_grad_norm),
        use_amp=bool(use_amp),
        # critic optimizer / scheduler config
        critic_optim_name=str(critic_optim_name),
        critic_lr=float(critic_lr),
        critic_weight_decay=float(critic_weight_decay),
        critic_sched_name=str(critic_sched_name),
        # scheduler shared knobs
        total_steps=int(total_steps),
        warmup_steps=int(warmup_steps),
        min_lr_ratio=float(min_lr_ratio),
        poly_power=float(poly_power),
        step_size=int(step_size),
        sched_gamma=float(sched_gamma),
        milestones=milestones_t,
    )

    # ---------------------------------------------------------------------
    # 3) Algorithm: rollout collection + GAE/returns + update scheduling
    # ---------------------------------------------------------------------
    # TRPO is typically full-batch. If minibatch_size is None, ensure your
    # OnPolicyAlgorithm treats it as "use full batch"; otherwise pass rollout_steps.
    algo = OnPolicyAlgorithm(
        head=head,
        core=core,
        rollout_steps=rollout_steps_i,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        update_epochs=update_epochs_i,
        minibatch_size=minibatch_size_i,
        device=device,
        dtype_obs=dtype_obs,
        dtype_act=dtype_act,
        normalize_advantages=bool(normalize_advantages),
        adv_eps=float(adv_eps),
    )
    return algo
