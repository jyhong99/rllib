"""Public API for discrete-action SAC baseline.

Notes
-----
This package intentionally exposes a compact import surface while allowing
internal module layout to evolve.

Exports
-------
sac_discrete
    Factory that builds a complete :class:`OffPolicyAlgorithm` instance for
    discrete SAC.
SACDiscreteHead
    Network container with actor, online critic, and target critic modules.
SACDiscreteCore
    Optimization engine implementing critic/actor/alpha updates and target sync.
"""

from __future__ import annotations

# Public builder / entrypoint
from rllib.model_free.baselines.policy_based.off_policy.sac_discrete.sac_discrete import sac_discrete

# Core components
from rllib.model_free.baselines.policy_based.off_policy.sac_discrete.head import SACDiscreteHead
from rllib.model_free.baselines.policy_based.off_policy.sac_discrete.core import SACDiscreteCore

__all__ = [
    "sac_discrete",
    "SACDiscreteHead",
    "SACDiscreteCore",
]
