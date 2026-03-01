"""Implicit Q-Learning (IQL) offline baseline package.

This package exposes the high-level IQL factory and its composable head/core
components. The composition follows the repository-wide pattern:

- ``Head``: neural modules and inference utilities.
- ``Core``: optimization/update logic.
- ``OffPolicyAlgorithm``: shared replay/update orchestration.
"""

from __future__ import annotations

from rllib.model_free.baselines.offline.iql.core import IQLCore
from rllib.model_free.baselines.offline.iql.head import IQLHead
from rllib.model_free.baselines.offline.iql.iql import iql

__all__ = [
    "iql",
    "IQLHead",
    "IQLCore",
]
