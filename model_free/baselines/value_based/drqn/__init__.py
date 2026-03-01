"""Public exports for the DRQN baseline package.

This module centralizes the DRQN user-facing API:

- :func:`drqn` for constructing the complete training algorithm wrapper.
- :class:`DRQNHead` for recurrent Q-network policy logic.
- :class:`DRQNCore` for sequence-aware temporal-difference optimization.

Notes
-----
Importing from this package is preferred over deep module imports when
consuming DRQN in external training scripts.
"""

from __future__ import annotations

from rllib.model_free.baselines.value_based.drqn.core import DRQNCore
from rllib.model_free.baselines.value_based.drqn.drqn import drqn
from rllib.model_free.baselines.value_based.drqn.head import DRQNHead

__all__ = [
    "drqn",
    "DRQNHead",
    "DRQNCore",
]
