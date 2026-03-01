"""IQN policy head.

This module defines the tau-conditioned quantile head used by IQN and includes
a worker reconstruction entrypoint for Ray-based rollouts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.q_networks import TauQuantileQNetwork
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_iqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build an IQN head on CPU for rollout workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe keyword arguments produced by
        :meth:`IQNHead._export_kwargs_json_safe`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training mode disabled where available.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = IQNHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class IQNHead(QLearningHead):
    """IQN head with online/target tau-conditioned quantile networks."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        n_cos_embeddings: int = 64,
        n_eval_quantile_samples: int = 32,
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
        """Initialize IQN head networks.

        Parameters
        ----------
        obs_dim : int
            Flattened observation dimension.
        n_actions : int
            Number of discrete actions.
        n_cos_embeddings : int, default=64
            Number of cosine basis embeddings for sampled taus.
        n_eval_quantile_samples : int, default=32
            Default number of tau samples for expected-Q evaluation.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden widths for quantile value head MLP.
        activation_fn : Any, default=torch.nn.ReLU
            Activation module class.
        dueling_mode : bool, default=False
            Enable dueling decomposition.
        obs_shape : tuple[int, ...] | None, default=None
            Optional original observation shape.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor kwargs.
        init_trunk : bool | None, default=None
            Optional control for external trunk initialization.
        init_type : str, default="orthogonal"
            Parameter initializer name.
        gain : float, default=1.0
            Initializer gain.
        bias : float, default=0.0
            Initializer bias.
        device : str | torch.device
            Device for module parameters and buffers.

        Raises
        ------
        ValueError
            If dimensions are invalid.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_cos_embeddings = int(n_cos_embeddings)
        self.n_eval_quantile_samples = int(n_eval_quantile_samples)
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
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")
        if self.n_cos_embeddings <= 0:
            raise ValueError(f"n_cos_embeddings must be > 0, got {self.n_cos_embeddings}")
        if self.n_eval_quantile_samples <= 0:
            raise ValueError(
                f"n_eval_quantile_samples must be > 0, got {self.n_eval_quantile_samples}"
            )
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive integers, got: {self.hidden_sizes}")

        q_kwargs: Dict[str, Any] = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "n_quantiles": self.n_eval_quantile_samples,
            "n_cos_embeddings": self.n_cos_embeddings,
            "hidden_sizes": self.hidden_sizes,
            "activation_fn": self.activation_fn,
            "dueling_mode": self.dueling_mode,
            "obs_shape": self.obs_shape,
            "feature_extractor_cls": self.feature_extractor_cls,
            "feature_extractor_kwargs": self.feature_extractor_kwargs,
            "init_trunk": self.init_trunk,
            "init_type": self.init_type,
            "gain": self.gain,
            "bias": self.bias,
        }

        def _build_quantile_net() -> TauQuantileQNetwork:
            """Create one tau-conditioned quantile network branch."""
            return TauQuantileQNetwork(**q_kwargs).to(self.device)

        self.q = _build_quantile_net()
        self.q_target = _build_quantile_net()

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    def sample_taus(self, batch_size: int, n_samples: int) -> th.Tensor:
        """Sample tau fractions uniformly from ``[0, 1)``.

        Parameters
        ----------
        batch_size : int
            Number of batch elements.
        n_samples : int
            Number of tau samples per batch element.

        Returns
        -------
        torch.Tensor
            Tau samples with shape ``(B, N)``.

        Raises
        ------
        ValueError
            If ``batch_size`` or ``n_samples`` is non-positive.
        """
        bsz = int(batch_size)
        ns = int(n_samples)
        if bsz <= 0:
            raise ValueError(f"batch_size must be > 0, got {bsz}")
        if ns <= 0:
            raise ValueError(f"n_samples must be > 0, got {ns}")
        return th.rand((bsz, ns), device=self.device)

    def quantiles(self, obs: Any, taus: th.Tensor) -> th.Tensor:
        """Evaluate online quantile action-values at given taus.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.
        taus : torch.Tensor
            Tau samples with shape ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Quantile action-values of shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        tau = taus.to(device=self.device)
        return self.q(s, tau)

    @th.no_grad()
    def quantiles_target(self, obs: Any, taus: th.Tensor) -> th.Tensor:
        """Evaluate target quantile action-values at given taus.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.
        taus : torch.Tensor
            Tau samples with shape ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Target quantile action-values of shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        tau = taus.to(device=self.device)
        return self.q_target(s, tau)

    @staticmethod
    def q_mean_from_quantiles(quantiles: th.Tensor) -> th.Tensor:
        """Reduce quantile values to expected Q-values.

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
            If ``quantiles`` is not a rank-3 tensor.
        """
        if quantiles.dim() != 3:
            raise ValueError(f"Expected quantiles shape (B,N,A), got: {tuple(quantiles.shape)}")
        return quantiles.mean(dim=1)

    def q_values(self, obs: Any, n_samples: int | None = None) -> th.Tensor:
        """Estimate online expected Q-values by Monte Carlo tau sampling.

        Parameters
        ----------
        obs : Any
            Observation batch.
        n_samples : int | None, default=None
            Number of tau samples. Uses ``self.n_eval_quantile_samples`` when
            ``None``.

        Returns
        -------
        torch.Tensor
            Expected Q-values of shape ``(B, A)``.
        """
        s = self._to_tensor_batched(obs)
        n = self.n_eval_quantile_samples if n_samples is None else int(n_samples)
        taus = self.sample_taus(batch_size=int(s.shape[0]), n_samples=n)
        z = self.q(s, taus)
        return self.q_mean_from_quantiles(z)

    @th.no_grad()
    def q_values_target(self, obs: Any, n_samples: int | None = None) -> th.Tensor:
        """Estimate target expected Q-values by Monte Carlo tau sampling.

        Parameters
        ----------
        obs : Any
            Observation batch.
        n_samples : int | None, default=None
            Number of tau samples. Uses ``self.n_eval_quantile_samples`` when
            ``None``.

        Returns
        -------
        torch.Tensor
            Target expected Q-values of shape ``(B, A)``.
        """
        s = self._to_tensor_batched(obs)
        n = self.n_eval_quantile_samples if n_samples is None else int(n_samples)
        taus = self.sample_taus(batch_size=int(s.shape[0]), n_samples=n)
        z = self.q_target(s, taus)
        return self.q_mean_from_quantiles(z)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe format.

        Returns
        -------
        Dict[str, Any]
            Constructor payload compatible with checkpoint metadata and Ray
            worker reconstruction.
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
            "n_cos_embeddings": int(self.n_cos_embeddings),
            "n_eval_quantile_samples": int(self.n_eval_quantile_samples),
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
        """Save IQN head checkpoint.

        Parameters
        ----------
        path : str
            Destination checkpoint path. ``.pt`` is appended if missing.
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
        """Load IQN head checkpoint.

        Parameters
        ----------
        path : str
            Path to a checkpoint created by :meth:`save`.

        Raises
        ------
        ValueError
            If the checkpoint format is invalid.
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
        """Get Ray worker policy factory specification.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint and JSON-safe kwargs for reconstructing this head on
            rollout workers.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_iqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
