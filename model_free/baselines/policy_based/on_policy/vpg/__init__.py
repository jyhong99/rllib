"""VPG package exports.

This subpackage exposes the public API for continuous-action VPG baselines.

Attributes
----------
vpg : Callable[..., OnPolicyAlgorithm]
    High-level builder that composes :class:`VPGHead`, :class:`VPGCore`, and
    :class:`OnPolicyAlgorithm`.
VPGHead : type
    Continuous actor network with optional state-value baseline critic.
VPGCore : type
    Policy-gradient update core with optional baseline regression.

Notes
-----
- This package targets continuous control.
- Baseline usage is controlled by head configuration and enforced in the core.
- ``__all__`` defines the stable import surface.

Examples
--------
Import the builder::

    from rllib.model_free.baselines.policy_based.on_policy.vpg import vpg

Import all public symbols::

    from rllib.model_free.baselines.policy_based.on_policy.vpg import VPGCore, VPGHead, vpg
"""

from __future__ import annotations

from rllib.model_free.baselines.policy_based.on_policy.vpg.vpg import vpg
from rllib.model_free.baselines.policy_based.on_policy.vpg.head import VPGHead
from rllib.model_free.baselines.policy_based.on_policy.vpg.core import VPGCore

__all__ = [
    "vpg",
    "VPGHead",
    "VPGCore",
]
