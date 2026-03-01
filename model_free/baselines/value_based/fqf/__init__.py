"""Public exports for the FQF baseline package.

This package exposes:

- :func:`fqf` for constructing the full algorithm wrapper.
- :class:`FQFHead` for model and action-value logic.
- :class:`FQFCore` for optimization logic.
"""

from __future__ import annotations

from rllib.model_free.baselines.value_based.fqf.core import FQFCore
from rllib.model_free.baselines.value_based.fqf.fqf import fqf
from rllib.model_free.baselines.value_based.fqf.head import FQFHead

__all__ = [
    "fqf",
    "FQFHead",
    "FQFCore",
]
