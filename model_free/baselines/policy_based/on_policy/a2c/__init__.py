"""
A2C
=======

This subpackage provides the components required to build and train an
Advantage Actor-Critic (A2C) agent for **continuous** action spaces.

Public API
----------
a2c : callable
    High-level builder that wires together the A2C head, core, and the
    :class:`model_free.common.policies.on_policy_algorithm.OnPolicyAlgorithm`.

A2CHead : torch.nn.Module
    Actor-critic network container:
    - actor: diagonal Gaussian policy network
    - critic: state-value network V(s)

A2CCore : ActorCriticCore
    Update engine implementing one A2C optimization step:
    - policy/value/entropy losses
    - optimizer steps (actor + critic)
    - optional AMP
    - global gradient clipping
    - optional learning-rate schedulers (via the base core)

Notes
-----
- This implementation is **continuous-only**. For discrete action spaces, use a
  categorical policy head/core pair.
- Only the symbols listed in ``__all__`` are considered part of the stable,
  public import surface of this package.

Examples
--------
Construct an A2C algorithm instance::

    from rllib.model_free.algos.a2c import a2c

    algo = a2c(obs_dim=obs_dim, action_dim=action_dim, device="cuda:0")

Or import individual components::

    from rllib.model_free.algos.a2c import A2CHead, A2CCore
"""

from __future__ import annotations

# Re-export public symbols for a clean user-facing import surface.
from rllib.model_free.baselines.policy_based.on_policy.a2c.a2c import a2c
from rllib.model_free.baselines.policy_based.on_policy.a2c.core import A2CCore
from rllib.model_free.baselines.policy_based.on_policy.a2c.head import A2CHead

__all__ = [
    "a2c",
    "A2CHead",
    "A2CCore",
]
