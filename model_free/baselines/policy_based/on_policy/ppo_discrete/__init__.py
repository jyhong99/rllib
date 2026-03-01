"""Discrete PPO package exports.

This subpackage exposes the public PPO API for discrete action spaces.

Attributes
----------
ppo_discrete : Callable[..., OnPolicyAlgorithm]
    High-level builder that composes :class:`PPODiscreteHead`,
    :class:`PPODiscreteCore`, and :class:`OnPolicyAlgorithm`.
PPODiscreteHead : type
    Discrete actor-critic head with categorical policy and state-value critic.
PPODiscreteCore : type
    PPO minibatch update core for clipped surrogate optimization.

Notes
-----
- This package targets discrete control problems.
- PPO-specific clipping/KL/value-loss behavior is implemented in the core.
- ``__all__`` defines the stable import surface for callers.

Examples
--------
Import the builder::

    from rllib.model_free.baselines.policy_based.on_policy.ppo_discrete import ppo_discrete

Import all public symbols::

    from rllib.model_free.baselines.policy_based.on_policy.ppo_discrete import PPODiscreteCore, PPODiscreteHead, ppo_discrete
"""

from __future__ import annotations

from rllib.model_free.baselines.policy_based.on_policy.ppo_discrete.core import PPODiscreteCore
from rllib.model_free.baselines.policy_based.on_policy.ppo_discrete.head import PPODiscreteHead
from rllib.model_free.baselines.policy_based.on_policy.ppo_discrete.ppo_discrete import ppo_discrete

__all__ = [
    "ppo_discrete",
    "PPODiscreteHead",
    "PPODiscreteCore",
]
