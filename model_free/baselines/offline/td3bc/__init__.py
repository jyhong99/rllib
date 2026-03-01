"""TD3+BC offline baseline package.

This package exposes:

- ``td3bc``: high-level factory that builds a complete offline TD3+BC algorithm.
- ``TD3BCHead``: deterministic actor/twin-critic network container.
- ``TD3BCCore``: update logic combining TD3 critic learning with a behavior
  cloning regularized actor objective.
"""

from __future__ import annotations

from rllib.model_free.baselines.offline.td3bc.core import TD3BCCore
from rllib.model_free.baselines.offline.td3bc.head import TD3BCHead
from rllib.model_free.baselines.offline.td3bc.td3bc import td3bc

__all__ = [
    "td3bc",
    "TD3BCHead",
    "TD3BCCore",
]
