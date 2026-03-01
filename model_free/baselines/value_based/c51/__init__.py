"""C51 package exports.

This subpackage exposes the public API for the C51 baseline.

Attributes
----------
c51 : Callable[..., OffPolicyAlgorithm]
    High-level builder that composes :class:`C51Head`, :class:`C51Core`, and
    :class:`OffPolicyAlgorithm`.
C51Head : type
    Distributional Q-network head with online/target categorical critics.
C51Core : type
    Categorical Bellman projection update core with target-network handling.
"""

from __future__ import annotations

from rllib.model_free.baselines.value_based.c51.c51 import c51
from rllib.model_free.baselines.value_based.c51.core import C51Core
from rllib.model_free.baselines.value_based.c51.head import C51Head

__all__ = [
    "c51",
    "C51Head",
    "C51Core",
]
