"""Public exports for the Rainbow baseline package.

This package exposes:

- :func:`rainbow` for constructing the complete algorithm wrapper.
- :class:`RainbowHead` for distributional policy/value networks.
- :class:`RainbowCore` for C51 optimization logic.
"""

from __future__ import annotations

from rllib.model_free.baselines.value_based.rainbow.core import RainbowCore
from rllib.model_free.baselines.value_based.rainbow.head import RainbowHead
from rllib.model_free.baselines.value_based.rainbow.rainbow import rainbow

__all__ = [
    "rainbow",
    "RainbowHead",
    "RainbowCore",
]
