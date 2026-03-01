"""C51 Head.

This module defines the distributional Q-network head used by C51 and its Ray
worker reconstruction helper.

It provides:

- :func:`build_c51_head_worker_policy` for CPU-only worker rebuild.
- :class:`C51Head` with online/target categorical Q-networks on a fixed support.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.q_networks import RainbowQNetwork
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_c51_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU-only C51 head on Ray workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe constructor kwargs for :class:`C51Head`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training disabled where supported.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = C51Head(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class C51Head(QLearningHead):
    """C51 head with online/target categorical Q-networks.

    Notes
    -----
    - Uses a fixed support ``linspace(v_min, v_max, atom_size)``.
    - ``q`` and ``q_target`` are initialized identically, then target is frozen.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize online/target distributional Q-networks for C51.

        Parameters
        ----------
        obs_dim : int
            Observation vector dimension.
        n_actions : int
            Number of discrete actions.
        atom_size : int, default=51
            Number of categorical atoms per action.
        v_min : float, default=-10.0
            Minimum support value.
        v_max : float, default=10.0
            Maximum support value.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer sizes for Q-networks.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class.
        obs_shape : tuple[int, ...] | None, default=None
            Optional raw observation shape for feature extractor building.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor kwargs.
        init_trunk : bool | None, default=None
            Optional trunk-init toggle forwarded to network builders.
        init_type : str, default="orthogonal"
            Initialization scheme identifier.
        gain : float, default=1.0
            Initialization gain.
        bias : float, default=0.0
            Initialization bias value.
        noisy_std_init : float, default=0.0
            Initial noise std for noisy layers (if enabled by network impl).
        device : str | torch.device, default=cuda if available else cpu
            Module device.

        Raises
        ------
        ValueError
            If action/atom/support configuration is invalid.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.atom_size = int(atom_size)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.noisy_std_init = float(noisy_std_init)

        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")
        if self.atom_size <= 1:
            raise ValueError(f"atom_size must be >= 2, got {self.atom_size}")
        if not (self.v_min < self.v_max):
            raise ValueError(f"Require v_min < v_max, got v_min={self.v_min}, v_max={self.v_max}")

        support = th.linspace(self.v_min, self.v_max, self.atom_size, dtype=th.float32)
        self.register_buffer("support", support.detach().clone())
        support_q = self.support.detach().clone()
        support_t = self.support.detach().clone()

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build one feature-extractor pair for a Q-network branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        fe_q, fd_q = _build_fe()
        fe_t, fd_t = _build_fe()

        self.q = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_q,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor=fe_q,
            feature_dim=fd_q,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        self.q_target = RainbowQNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            atom_size=self.atom_size,
            support=support_t,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor=fe_t,
            feature_dim=fd_t,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
            noisy_std_init=self.noisy_std_init,
        ).to(self.device)

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe form.

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor payload for checkpoints and Ray workers.
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
            "atom_size": int(self.atom_size),
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": fe_name,
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "noisy_std_init": float(self.noisy_std_init),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """Save head weights and metadata to disk.

        Parameters
        ----------
        path : str
            Output checkpoint path. ``.pt`` is appended if missing.
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "support": self.support.detach().cpu(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """Load checkpoint weights into the existing instance.

        Parameters
        ----------
        path : str
            Checkpoint path created by :meth:`save`.

        Raises
        ------
        ValueError
            If checkpoint format is not recognized.
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.q.load_state_dict(ckpt["q"])

        if ckpt.get("q_target", None) is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
            self.freeze_target(self.q_target)
            self.q_target.eval()
        else:
            self.hard_update(self.q_target, self.q)
            self.freeze_target(self.q_target)

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Build a Ray-safe factory spec for worker reconstruction.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint + JSON-safe kwargs used to rebuild this head remotely.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_c51_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
