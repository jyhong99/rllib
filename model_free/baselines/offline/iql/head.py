"""Network head definitions for offline Implicit Q-Learning (IQL).

This module contains:

- a worker-safe construction function for distributed reconstruction
- :class:`IQLHead`, which bundles actor, twin critics, target critics, and a
  separate state-value network required by IQL updates

The head is intentionally decoupled from update logic so the same modules can be
used by shared off-policy training wrappers.
"""


from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from rllib.model_free.common.networks.value_networks import DoubleStateActionValueNetwork, StateValueNetwork
from rllib.model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_iql_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU-based IQL head instance for worker processes.

    Parameters
    ----------
    **kwargs : Any
        Serialized constructor arguments originally exported by ``IQLHead``.

    Returns
    -------
    nn.Module
        ``IQLHead`` instance configured for CPU execution.

    Notes
    -----
    This helper resolves serialized callables (for example activation functions
    and feature extractor classes), forces ``device="cpu"``, and disables
    training mode when the head exposes ``set_training``.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = IQLHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class IQLHead(OffPolicyContinuousActorCriticHead):
    """IQL network container for continuous-action offline learning.

    The head owns all neural modules needed by IQL:

    - stochastic actor (Gaussian policy with tanh squashing)
    - twin Q critics for clipped value estimation
    - target twin critics for stable bootstrapping
    - standalone state-value network for expectile regression
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
    ) -> None:
        """Initialize actor/critic/value modules for IQL.

        Parameters
        ----------
        obs_dim : int
            Flattened observation feature dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer sizes shared by actor/critic/value MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation module/callable for network blocks.
        init_type : str, default="orthogonal"
            Parameter initialization strategy.
        gain : float, default=1.0
            Gain parameter for compatible initializers.
        bias : float, default=0.0
            Constant bias initialization value.
        obs_shape : tuple[int, ...], optional
            Original observation shape used by optional feature extractors.
        feature_extractor_cls : type[torch.nn.Module], optional
            Optional encoder class instantiated by ``build_feature_extractor``.
        feature_extractor_kwargs : dict[str, Any], optional
            Keyword arguments for ``feature_extractor_cls``.
        init_trunk : bool or None, optional
            Controls whether trunk/extractor modules are explicitly initialized.
        device : str or torch.device, default=auto
            Device to place created modules on.
        log_std_mode : str, default="layer"
            Policy log-standard-deviation parameterization mode.
        log_std_init : float, default=-0.5
            Initial policy log-standard-deviation value.

        Returns
        -------
        None
            Initializes module members in place.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        fe_a, fd_a = build_feature_extractor(
            obs_dim=self.obs_dim,
            obs_shape=self.obs_shape,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
        )
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=True,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            feature_extractor=fe_a,
            feature_dim=fd_a,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.critic = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            obs_shape=self.obs_shape,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.critic_target = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            obs_shape=self.obs_shape,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        fe_v, fd_v = build_feature_extractor(
            obs_dim=self.obs_dim,
            obs_shape=self.obs_shape,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
        )
        self.value = StateValueNetwork(
            state_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor=fe_v,
            feature_dim=fd_v,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    def value_values(self, obs: Any) -> th.Tensor:
        """Compute state-value predictions for a batch of observations.

        Parameters
        ----------
        obs : Any
            Observation input accepted by ``_to_tensor_batched``.

        Returns
        -------
        torch.Tensor
            Value estimates from ``self.value`` with shape ``(B, 1)``.
        """
        s = self._to_tensor_batched(obs)
        return self.value(s)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor settings in a JSON-safe format.

        Returns
        -------
        dict[str, Any]
            Serialized constructor arguments suitable for worker reconstruction
            and checkpoint metadata.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": (
                self.feature_extractor_cls
                if isinstance(self.feature_extractor_cls, str)
                else (
                    getattr(self.feature_extractor_cls, "__name__", None)
                    if self.feature_extractor_cls is not None
                    else None
                )
            ),
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "device": str(self.device),
            "log_std_mode": str(self.log_std_mode),
            "log_std_init": float(self.log_std_init),
        }

    def save(self, path: str) -> None:
        """Persist IQL head parameters and construction metadata to disk.

        Parameters
        ----------
        path : str
            Target checkpoint path. ``.pt`` is appended when missing.

        Returns
        -------
        None
            Writes checkpoint content to ``path``.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value": self.value.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """Load model parameters from an IQL head checkpoint.

        Parameters
        ----------
        path : str
            Source checkpoint path. ``.pt`` is appended when missing.

        Returns
        -------
        None
            Restores actor/critic/value parameters in place.

        Raises
        ------
        ValueError
            If checkpoint format does not match expected IQL payload keys.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.freeze_target(self.critic_target)
            self.critic_target.eval()
        else:
            self.hard_update(self.critic_target, self.critic)
            self.freeze_target(self.critic_target)

        if ckpt.get("value", None) is not None:
            self.value.load_state_dict(ckpt["value"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Create a Ray worker policy reconstruction specification.

        Returns
        -------
        PolicyFactorySpec
            Serializable factory spec containing the worker entrypoint and
            JSON-safe constructor kwargs.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_iql_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
