"""QR-DQN policy head.

This module provides the fixed-quantile head used by QR-DQN and a Ray worker
factory entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.q_networks import FixedQuantileQNetwork
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_qrdqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU QR-DQN head for Ray rollout workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe constructor payload produced by
        :meth:`QRDQNHead._export_kwargs_json_safe`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training mode disabled where supported.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = QRDQNHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class QRDQNHead(QLearningHead):
    """QR-DQN head with online and target fixed-quantile Q-networks."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 200,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize QR-DQN head modules.

        Parameters
        ----------
        obs_dim : int
            Flattened observation dimension.
        n_actions : int
            Number of discrete actions.
        n_quantiles : int, default=200
            Number of fixed quantiles per action.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden widths for quantile network MLP.
        activation_fn : Any, default=torch.nn.ReLU
            Activation module class.
        dueling_mode : bool, default=False
            Enable dueling decomposition in quantile network heads.
        obs_shape : tuple[int, ...] | None, default=None
            Optional raw observation shape.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor keyword arguments.
        init_trunk : bool | None, default=None
            Control whether external trunks are initialized.
        init_type : str, default="orthogonal"
            Initializer name.
        gain : float, default=1.0
            Initializer gain.
        bias : float, default=0.0
            Initializer bias value.
        device : str | torch.device
            Target device.

        Raises
        ------
        ValueError
            If dimensions are invalid.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.dueling_mode = bool(dueling_mode)
        self.obs_shape = obs_shape
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.init_trunk = init_trunk
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got: {self.obs_dim}")
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got: {self.n_actions}")
        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got: {self.n_quantiles}")
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive integers, got: {self.hidden_sizes}")

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build one feature extractor pair for one network branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        q_kwargs: Dict[str, Any] = {
            "state_dim": self.obs_dim,
            "action_dim": self.n_actions,
            "n_quantiles": self.n_quantiles,
            "hidden_sizes": self.hidden_sizes,
            "activation_fn": self.activation_fn,
            "dueling_mode": self.dueling_mode,
            "init_trunk": self.init_trunk,
            "init_type": self.init_type,
            "gain": self.gain,
            "bias": self.bias,
        }

        def _build_q_branch() -> FixedQuantileQNetwork:
            """Create one fixed-quantile branch with its own feature extractor."""
            fe, fd = _build_fe()
            return FixedQuantileQNetwork(
                feature_extractor=fe,
                feature_dim=fd,
                **q_kwargs,
            ).to(self.device)

        self.q = _build_q_branch()
        self.q_target = _build_q_branch()

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    def quantiles(self, obs: Any) -> th.Tensor:
        """Compute online quantiles ``Z(s, a)``.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.

        Returns
        -------
        torch.Tensor
            Quantile tensor with shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        return self.q(s)

    @th.no_grad()
    def quantiles_target(self, obs: Any) -> th.Tensor:
        """Compute target quantiles ``Z_target(s, a)``.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.

        Returns
        -------
        torch.Tensor
            Target quantile tensor with shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        return self.q_target(s)

    @staticmethod
    def q_mean_from_quantiles(quantiles: th.Tensor) -> th.Tensor:
        """Convert quantiles to expected Q-values.

        Parameters
        ----------
        quantiles : torch.Tensor
            Quantile tensor with shape ``(B, N, A)``.

        Returns
        -------
        torch.Tensor
            Expected Q-values with shape ``(B, A)``.

        Raises
        ------
        ValueError
            If ``quantiles`` is not rank-3.
        """
        if quantiles.dim() != 3:
            raise ValueError(f"Expected quantiles shape (B,N,A), got: {tuple(quantiles.shape)}")
        return quantiles.mean(dim=1)

    def q_values(self, obs: Any) -> th.Tensor:
        """Compute expected online Q-values.

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        torch.Tensor
            Expected Q-values with shape ``(B, A)``.
        """
        return self.q_mean_from_quantiles(self.quantiles(obs))

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        """Compute expected target Q-values.

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        torch.Tensor
            Expected target Q-values with shape ``(B, A)``.
        """
        return self.q_mean_from_quantiles(self.quantiles_target(obs))

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe format.

        Returns
        -------
        Dict[str, Any]
            Keyword payload used for checkpoints and Ray worker rebuilds.
        """
        fe_cls = self.feature_extractor_cls
        if isinstance(fe_cls, str):
            fe_name: Optional[str] = fe_cls
        elif fe_cls is not None:
            fe_name = getattr(fe_cls, "__name__", None) or str(fe_cls)
        else:
            fe_name = None

        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "n_quantiles": int(self.n_quantiles),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": fe_name,
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """Save head checkpoint.

        Parameters
        ----------
        path : str
            Destination path. ``.pt`` is appended if missing.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """Load head checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint produced by :meth:`save`.

        Raises
        ------
        ValueError
            If checkpoint format is invalid.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.q.load_state_dict(ckpt["q"])

        if ckpt.get("q_target", None) is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
        else:
            self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)
        self.q_target.eval()

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Get Ray worker reconstruction specification.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint and JSON-safe kwargs for worker-side policy rebuild.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_qrdqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
