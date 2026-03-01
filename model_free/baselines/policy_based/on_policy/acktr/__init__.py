"""ACKTR package exports.

This subpackage exposes the public ACKTR baseline surface for continuous action
spaces.

Attributes
----------
acktr : Callable[..., OnPolicyAlgorithm]
    High-level builder that composes an :class:`ACKTRHead`,
    :class:`ACKTRCore`, and :class:`OnPolicyAlgorithm`.
ACKTRHead : type
    Continuous actor-critic head with Gaussian policy and state-value critic.
ACKTRCore : type
    Update core that applies A2C-style losses with K-FAC-oriented optimizer
    configuration.

Notes
-----
- This package targets continuous control only.
- K-FAC/trust-region behavior is implemented in the core/optimizer path rather
  than the network container itself.
- ``__all__`` defines the stable import surface for callers.

Examples
--------
Import the builder::

    from rllib.model_free.baselines.policy_based.on_policy.acktr import acktr

Import all public symbols::

    from rllib.model_free.baselines.policy_based.on_policy.acktr import ACKTRCore, ACKTRHead, acktr
"""

from __future__ import annotations

# Re-export public symbols for a clean user-facing import surface.
from rllib.model_free.baselines.policy_based.on_policy.acktr.acktr import acktr
from rllib.model_free.baselines.policy_based.on_policy.acktr.core import ACKTRCore
from rllib.model_free.baselines.policy_based.on_policy.acktr.head import ACKTRHead

__all__ = [
    "acktr",
    "ACKTRHead",
    "ACKTRCore",
]
