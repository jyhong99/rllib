"""DQN package exports.

This subpackage exposes the public API for DQN baselines.

Attributes
----------
dqn : Callable[..., OffPolicyAlgorithm]
    High-level builder that composes :class:`DQNHead`, :class:`DQNCore`, and
    :class:`OffPolicyAlgorithm`.
DQNHead : type
    Discrete Q-network head with online and target critics.
DQNCore : type
    TD update core supporting vanilla and Double DQN behavior.
"""

from __future__ import annotations

from rllib.model_free.baselines.value_based.dqn.core import DQNCore
from rllib.model_free.baselines.value_based.dqn.dqn import dqn
from rllib.model_free.baselines.value_based.dqn.head import DQNHead

__all__ = [
    "dqn",
    "DQNHead",
    "DQNCore",
]
