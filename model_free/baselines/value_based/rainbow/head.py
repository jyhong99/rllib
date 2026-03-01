"""Rainbow policy head.

This module defines the distributional Rainbow head (C51 + NoisyNet-ready
network) and a Ray worker reconstruction entrypoint.
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


def build_rainbow_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU Rainbow head for Ray rollout workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe constructor payload produced by
        :meth:`RainbowHead._export_kwargs_json_safe`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training mode disabled where supported.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = RainbowHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class RainbowHead(QLearningHead):
    """Rainbow head with online/target categorical Q-networks."""

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
        noisy_std_init: float = 0.5,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize Rainbow head.

        Parameters
        ----------
        obs_dim : int
            Flattened observation dimension.
        n_actions : int
            Number of discrete actions.
        atom_size : int, default=51
            Number of C51 support atoms (>= 2).
        v_min : float, default=-10.0
            Lower support bound.
        v_max : float, default=10.0
            Upper support bound.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden widths for network trunk.
        activation_fn : Any, default=torch.nn.ReLU
            Activation module class.
        obs_shape : tuple[int, ...] | None, default=None
            Optional raw observation shape.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor kwargs.
        init_trunk : bool | None, default=None
            Control whether external trunks are initialized.
        init_type : str, default="orthogonal"
            Initializer name.
        gain : float, default=1.0
            Initializer gain.
        bias : float, default=0.0
            Initializer bias value.
        noisy_std_init : float, default=0.5
            Initial NoisyNet std.
        device : str | torch.device
            Target device.

        Raises
        ------
        ValueError
            If dimensions/support bounds are invalid.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.atom_size = int(atom_size)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.obs_shape = obs_shape
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.init_trunk = init_trunk
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.noisy_std_init = float(noisy_std_init)

        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")
        if self.atom_size < 2:
            raise ValueError(f"atom_size must be >= 2, got {self.atom_size}")
        if not (self.v_min < self.v_max):
            raise ValueError(f"Require v_min < v_max, got v_min={self.v_min}, v_max={self.v_max}")
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive integers, got: {self.hidden_sizes}")

        support = th.linspace(self.v_min, self.v_max, self.atom_size, dtype=th.float32)
        self.register_buffer("support", support.detach().clone())

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build one feature extractor pair for one distribution branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        q_kwargs: Dict[str, Any] = {
            "state_dim": self.obs_dim,
            "action_dim": self.n_actions,
            "atom_size": self.atom_size,
            "hidden_sizes": self.hidden_sizes,
            "activation_fn": self.activation_fn,
            "init_trunk": self.init_trunk,
            "init_type": self.init_type,
            "gain": self.gain,
            "bias": self.bias,
            "noisy_std_init": self.noisy_std_init,
        }

        def _build_q_branch(support_tensor: th.Tensor) -> RainbowQNetwork:
            """Create one Rainbow categorical branch."""
            fe, fd = _build_fe()
            return RainbowQNetwork(
                feature_extractor=fe,
                feature_dim=fd,
                support=support_tensor,
                **q_kwargs,
            ).to(self.device)

        self.q = _build_q_branch(self.support.detach().clone())
        self.q_target = _build_q_branch(self.support.detach().clone())

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    def reset_noise(self) -> None:
        """Resample NoisyNet noise buffers when supported."""
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()
        if hasattr(self.q_target, "reset_noise"):
            self.q_target.reset_noise()

    @th.no_grad()
    def act(self, obs: Any, *, epsilon: float = 0.0, deterministic: bool = True) -> th.Tensor:
        """Select action with NoisyNet refresh and epsilon-greedy policy.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.
        epsilon : float, default=0.0
            Random-action probability for epsilon-greedy.
        deterministic : bool, default=True
            If ``True``, uses greedy action selection aside from epsilon branch.

        Returns
        -------
        torch.Tensor
            Action indices.
        """
        if hasattr(self.q, "reset_noise"):
            self.q.reset_noise()
        return super().act(obs, epsilon=epsilon, deterministic=deterministic)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe format.

        Returns
        -------
        Dict[str, Any]
            Constructor payload for checkpoints and Ray worker reconstruction.
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
        """Save Rainbow head checkpoint.

        Parameters
        ----------
        path : str
            Destination path. ``.pt`` is appended if missing.
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
        """Load Rainbow head checkpoint.

        Parameters
        ----------
        path : str
            Path to a checkpoint produced by :meth:`save`.

        Raises
        ------
        ValueError
            If checkpoint format is invalid.
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"

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
        """Get Ray worker policy reconstruction specification.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint and JSON-safe kwargs for worker-side rebuild.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_rainbow_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
