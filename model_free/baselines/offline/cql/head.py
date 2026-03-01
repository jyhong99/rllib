"""Network head for Conservative Q-Learning (CQL).

This module defines the policy/critic network container used by the offline CQL
algorithm. The implementation intentionally subclasses the SAC continuous head so
that architecture and inference behavior remain consistent across SAC and CQL.

Notes
-----
CQL differs from SAC mainly in the critic objective (conservative penalty). The
network topology itself is identical in this codebase, so this head focuses on
construction and distributed-worker reconstruction hooks.
"""


from __future__ import annotations

from typing import Any

import torch.nn as nn

from rllib.model_free.baselines.policy_based.off_policy.sac.head import SACHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


class CQLHead(SACHead):
    """CQL network container for continuous-action control.

    The class inherits from :class:`~rllib.model_free.baselines.policy_based.off_policy.sac.head.SACHead`
    and therefore provides:

    - A squashed Gaussian actor for stochastic action sampling.
    - Twin Q critics for clipped double-Q learning.
    - Target critics for stable bootstrapped updates.

    Notes
    -----
    No CQL-specific architectural changes are introduced here. CQL-specific
    behavior is implemented in :class:`~rllib.model_free.baselines.offline.cql.core.CQLCore`.
    """

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Return a Ray reconstruction spec for this head instance.

        Returns
        -------
        PolicyFactorySpec
            Serializable factory metadata containing:

            - ``entrypoint``: importable callable used on Ray workers.
            - ``kwargs``: JSON-safe constructor arguments exported from this head.

        Notes
        -----
        This allows remote workers to rebuild an equivalent policy module with
        CPU-safe defaults for rollout/evaluation processes.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_cql_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )


def build_cql_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU policy module for Ray workers.

    Parameters
    ----------
    **kwargs : Any
        Serialized constructor arguments exported from ``CQLHead``.

    Returns
    -------
    nn.Module
        A ``CQLHead`` instance configured for worker-side inference on CPU.

    Notes
    -----
    The factory resolves potentially serialized callables such as
    ``activation_fn`` and ``feature_extractor_cls``, then disables training mode
    (if supported) to prevent worker-side stochastic state changes.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = CQLHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head
