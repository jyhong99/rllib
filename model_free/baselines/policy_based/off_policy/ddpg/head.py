"""DDPG head definitions.

This module reuses the TD3 head implementation because the model topology for
DDPG and TD3 is nearly identical in this codebase; DDPG behavior differences
are handled in the core update logic.
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


class DDPGHead(TD3Head):
    """DDPG head implemented as a thin TD3Head specialization.

    Notes
    -----
    - Uses the same actor/critic/target network construction as :class:`TD3Head`.
    - DDPG-vs-TD3 behavioral differences are expressed in core update rules
      (noise and delay), not in the network architecture here.
    """

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Return a Ray-safe factory spec for worker-side policy reconstruction.

        Parameters
        ----------
        None

        Returns
        -------
        PolicyFactorySpec
            Serializable policy-factory specification. The entrypoint points to
            :func:`build_ddpg_head_worker_policy` and kwargs are JSON-safe copies
            of this head's initialization arguments.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_ddpg_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )


def build_ddpg_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Construct a CPU DDPG head for distributed worker processes.

    Parameters
    ----------
    **kwargs : Any
        Serialized keyword arguments originally exported from a training-side
        :class:`DDPGHead`. Expected keys include network dimensions, activation
        identifiers, feature extractor metadata, and optional action bounds.

    Returns
    -------
    nn.Module
        A CPU-resident :class:`DDPGHead` instance configured in evaluation mode
        when ``set_training`` is available.

    Notes
    -----
    This helper performs robustness conversions before head construction:
    - resolves activation and feature-extractor entrypoints to callables/classes
    - converts action bounds to ``np.float32`` arrays for deterministic behavior
    - forces ``device='cpu'`` for Ray worker portability
    """
    kwargs = dict(kwargs)
    kwargs["device"] = "cpu"
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))
    kwargs["feature_extractor_cls"] = _resolve_feature_extractor_cls(
        kwargs.get("feature_extractor_cls", None)
    )

    if kwargs.get("action_low", None) is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    if kwargs.get("action_high", None) is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)

    head = DDPGHead(**kwargs).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head
