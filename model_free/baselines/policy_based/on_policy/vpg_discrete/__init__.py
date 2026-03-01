"""Discrete VPG package exports.

This subpackage exposes the public API for discrete-action VPG baselines.

Attributes
----------
vpg_discrete : Callable[..., OnPolicyAlgorithm]
    High-level builder that composes :class:`VPGDiscreteHead`,
    :class:`VPGDiscreteCore`, and :class:`OnPolicyAlgorithm`.
VPGDiscreteHead : type
    Discrete actor network with optional state-value baseline critic.
VPGDiscreteCore : type
    Policy-gradient update core with optional baseline regression.

Notes
-----
- This package targets discrete control.
- Baseline usage is controlled by head configuration and enforced in the core.
- ``__all__`` defines the stable import surface.

Examples
--------
Import the builder::

    from rllib.model_free.baselines.policy_based.on_policy.vpg_discrete import vpg_discrete

Import all public symbols::

    from rllib.model_free.baselines.policy_based.on_policy.vpg_discrete import VPGDiscreteCore, VPGDiscreteHead, vpg_discrete
"""

from __future__ import annotations

from rllib.model_free.baselines.policy_based.on_policy.vpg_discrete.vpg_discrete import vpg_discrete
from rllib.model_free.baselines.policy_based.on_policy.vpg_discrete.head import VPGDiscreteHead
from rllib.model_free.baselines.policy_based.on_policy.vpg_discrete.core import VPGDiscreteCore

__all__ = [
    "vpg_discrete",
    "VPGDiscreteHead",
    "VPGDiscreteCore",
]
