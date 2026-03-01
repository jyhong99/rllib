"""TRPO package exports.

This subpackage exposes the public API for continuous-action TRPO baselines.

Attributes
----------
trpo : Callable[..., OnPolicyAlgorithm]
    High-level builder that composes :class:`TRPOHead`, :class:`TRPOCore`, and
    :class:`OnPolicyAlgorithm`.
TRPOHead : type
    Continuous actor-critic head with Gaussian policy and state-value critic.
TRPOCore : type
    Trust-region update core implementing natural-gradient actor updates and
    critic regression.

Notes
-----
- This package targets continuous control.
- KL-constrained line search and CG/FVP logic live in the core.
- ``__all__`` defines the stable import surface.

Examples
--------
Import the builder::

    from rllib.model_free.baselines.policy_based.on_policy.trpo import trpo

Import all public symbols::

    from rllib.model_free.baselines.policy_based.on_policy.trpo import TRPOCore, TRPOHead, trpo
"""

from __future__ import annotations

from rllib.model_free.baselines.policy_based.on_policy.trpo.trpo import trpo
from rllib.model_free.baselines.policy_based.on_policy.trpo.head import TRPOHead
from rllib.model_free.baselines.policy_based.on_policy.trpo.core import TRPOCore

__all__ = [
    "trpo",
    "TRPOHead",
    "TRPOCore",
]
