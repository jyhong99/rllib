"""Conservative Q-Learning (CQL) offline baseline package.

This package exposes the high-level CQL builder and its composable head/core
components used by the common ``OffPolicyAlgorithm`` wrapper.

Modules
-------
cql
    User-facing factory function that constructs a fully wired CQL algorithm.
head
    Neural network head definition (actor, critics, target critics) reused from
    the SAC-style continuous-control architecture.
core
    Update logic implementing SAC-style actor-critic losses plus the CQL
    conservative regularization term.
"""

from __future__ import annotations

from rllib.model_free.baselines.offline.cql.core import CQLCore
from rllib.model_free.baselines.offline.cql.cql import cql
from rllib.model_free.baselines.offline.cql.head import CQLHead

__all__ = [
    "cql",
    "CQLHead",
    "CQLCore",
]
