"""Regularization helpers for pixel-based RL pipelines.

Exports
-------
DrQ
    - ``RandomShiftsAug``
    - ``drq_augment``
    - ``drq_augment_pair``

SVEA
    - ``RandomConvAug``
    - ``svea_augment``
    - ``svea_augment_pair``
    - ``svea_mix_loss``

Notes
-----
The symbols exported here are intentionally lightweight wrappers and utilities
that can be reused across baselines, policies, and trainer loops without
introducing algorithm-specific coupling.
"""

from __future__ import annotations

from rllib.model_free.common.regularizations.drq import RandomShiftsAug, drq_augment, drq_augment_pair
from rllib.model_free.common.regularizations.svea import (
    RandomConvAug,
    svea_augment,
    svea_augment_pair,
    svea_mix_loss,
)

__all__ = [
    "RandomShiftsAug",
    "drq_augment",
    "drq_augment_pair",
    "RandomConvAug",
    "svea_augment",
    "svea_augment_pair",
    "svea_mix_loss",
]
