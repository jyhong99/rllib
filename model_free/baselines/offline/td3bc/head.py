"""Network head definitions for offline TD3+BC.

The TD3+BC algorithm reuses the deterministic TD3 architecture:

- deterministic actor and target actor
- twin critics and target critics

This module adds TD3+BC-specific worker reconstruction helpers while leaving
network topology aligned with the existing TD3 implementation.
"""


from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from rllib.model_free.baselines.policy_based.off_policy.td3.head import TD3Head
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


class TD3BCHead(TD3Head):
    """TD3+BC network container for continuous control.

    Notes
    -----
    The class inherits from :class:`~rllib.model_free.baselines.policy_based.off_policy.td3.head.TD3Head`
    and introduces no architectural changes. TD3+BC-specific behavior is in the
    core loss function rather than in network structure.
    """

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Create Ray worker reconstruction metadata for this head.

        Returns
        -------
        PolicyFactorySpec
            Serializable policy factory specification with:

            - worker entrypoint callable
            - JSON-safe constructor kwargs exported from this head
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_td3bc_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )


def build_td3bc_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU TD3+BC head instance for distributed workers.

    Parameters
    ----------
    **kwargs : Any
        Serialized head constructor arguments.

    Returns
    -------
    nn.Module
        ``TD3BCHead`` instance configured for CPU-side inference/evaluation.

    Notes
    -----
    This helper resolves potentially string-serialized callables and normalizes
    action bounds into NumPy arrays when present.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    if cfg.get("action_low", None) is not None:
        cfg["action_low"] = np.asarray(cfg["action_low"], dtype=np.float32)
    if cfg.get("action_high", None) is not None:
        cfg["action_high"] = np.asarray(cfg["action_high"], dtype=np.float32)

    head = TD3BCHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head
