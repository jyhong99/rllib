"""FQF policy head.

This module defines the FQF head used by the off-policy driver and provides a
Ray worker reconstruction entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.q_networks import FractionProposalNetwork, TauQuantileQNetwork
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_fqf_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build an FQF head for CPU-only Ray workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe constructor payload produced by
        :meth:`FQFHead._export_kwargs_json_safe`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training mode disabled when supported.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = FQFHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class FQFHead(QLearningHead):
    """FQF head with quantile networks and fraction proposal network.

    Notes
    -----
    The head owns:

    - ``q``: online tau-conditioned quantile action-value network.
    - ``q_target``: target network used for bootstrapping.
    - ``fraction_net``: adaptive quantile fraction proposal network.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 32,
        n_cos_embeddings: int = 64,
        fraction_hidden_size: int = 128,
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
        """Initialize FQF online/target networks and fraction proposal network.

        Parameters
        ----------
        obs_dim : int
            Flattened observation dimension.
        n_actions : int
            Number of discrete actions.
        n_quantiles : int, default=32
            Number of quantile fractions.
        n_cos_embeddings : int, default=64
            Number of cosine basis embeddings for tau conditioning.
        fraction_hidden_size : int, default=128
            Hidden width of fraction proposal MLP.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden widths of quantile value head MLP.
        activation_fn : Any, default=torch.nn.ReLU
            Activation module class used by internal networks.
        dueling_mode : bool, default=False
            Whether quantile networks use dueling decomposition.
        obs_shape : tuple[int, ...] | None, default=None
            Optional original observation shape.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor keyword arguments.
        init_trunk : bool | None, default=None
            Controls trunk initialization behavior when external extractors are
            provided.
        init_type : str, default="orthogonal"
            Parameter initializer name.
        gain : float, default=1.0
            Initializer gain.
        bias : float, default=0.0
            Initializer bias constant.
        device : str | torch.device
            Device for all head modules.

        Raises
        ------
        ValueError
            If dimensions are invalid.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.n_cos_embeddings = int(n_cos_embeddings)
        self.fraction_hidden_size = int(fraction_hidden_size)
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
        if self.n_quantiles <= 1:
            raise ValueError(f"n_quantiles must be > 1 for FQF, got: {self.n_quantiles}")
        if self.n_cos_embeddings <= 0:
            raise ValueError(f"n_cos_embeddings must be > 0, got: {self.n_cos_embeddings}")
        if self.fraction_hidden_size <= 0:
            raise ValueError(
                f"fraction_hidden_size must be > 0, got: {self.fraction_hidden_size}"
            )
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive integers, got: {self.hidden_sizes}")

        q_kwargs: Dict[str, Any] = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "n_quantiles": self.n_quantiles,
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

        self.fraction_net = FractionProposalNetwork(
            feature_dim=int(self.q.feature_dim),
            n_quantiles=self.n_quantiles,
            hidden_size=self.fraction_hidden_size,
            activation_fn=self.activation_fn,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    def propose_fractions(self, obs: Any) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Propose adaptive quantile fractions for a batch of observations.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(taus, tau_hats, entropy)`` where:

            - ``taus`` has shape ``(B, N+1)``.
            - ``tau_hats`` has shape ``(B, N)``.
            - ``entropy`` has shape ``(B,)``.
        """
        s = self._to_tensor_batched(obs)
        state_feat = self.q.encode_state(s)
        return self.fraction_net(state_feat)

    def quantiles(self, obs: Any, tau_hats: th.Tensor) -> th.Tensor:
        """Evaluate online quantile values at given tau midpoints.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.
        tau_hats : torch.Tensor
            Tau midpoints with shape ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Quantile action-values of shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        tau = tau_hats.to(device=self.device)
        return self.q(s, tau)

    @th.no_grad()
    def quantiles_target(self, obs: Any, tau_hats: th.Tensor) -> th.Tensor:
        """Evaluate target quantile values at given tau midpoints.

        Parameters
        ----------
        obs : Any
            Observation batch accepted by :meth:`_to_tensor_batched`.
        tau_hats : torch.Tensor
            Tau midpoints with shape ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Target quantile action-values of shape ``(B, N, A)``.
        """
        s = self._to_tensor_batched(obs)
        tau = tau_hats.to(device=self.device)
        return self.q_target(s, tau)

    @staticmethod
    def q_mean_from_quantiles(quantiles: th.Tensor) -> th.Tensor:
        """Convert quantile distributions into expected Q-values.

        Parameters
        ----------
        quantiles : torch.Tensor
            Quantile tensor with shape ``(B, N, A)``.

        Returns
        -------
        torch.Tensor
            Mean Q-values with shape ``(B, A)``.

        Raises
        ------
        ValueError
            If ``quantiles`` is not rank-3.
        """
        if quantiles.dim() != 3:
            raise ValueError(f"Expected quantiles shape (B,N,A), got: {tuple(quantiles.shape)}")
        return quantiles.mean(dim=1)

    def q_values(self, obs: Any) -> th.Tensor:
        """Compute expected online Q-values by integrating over adaptive taus.

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        torch.Tensor
            Expected Q-values with shape ``(B, A)``.
        """
        _, tau_hats, _ = self.propose_fractions(obs)
        z = self.quantiles(obs, tau_hats=tau_hats.detach())
        return self.q_mean_from_quantiles(z)

    @th.no_grad()
    def q_values_target(self, obs: Any) -> th.Tensor:
        """Compute expected target Q-values by integrating over adaptive taus.

        Parameters
        ----------
        obs : Any
            Observation batch.

        Returns
        -------
        torch.Tensor
            Expected target Q-values with shape ``(B, A)``.
        """
        _, tau_hats, _ = self.propose_fractions(obs)
        z = self.quantiles_target(obs, tau_hats=tau_hats.detach())
        return self.q_mean_from_quantiles(z)

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe form.

        Returns
        -------
        Dict[str, Any]
            Keyword payload suitable for checkpoint metadata and Ray workers.
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
            "n_cos_embeddings": int(self.n_cos_embeddings),
            "fraction_hidden_size": int(self.fraction_hidden_size),
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
            Destination path. ``.pt`` is appended if omitted.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "fraction_net": self.fraction_net.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """Load head checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path produced by :meth:`save`.

        Raises
        ------
        ValueError
            If checkpoint structure is invalid.
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

        if ckpt.get("fraction_net", None) is not None:
            self.fraction_net.load_state_dict(ckpt["fraction_net"])

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Return Ray policy reconstruction spec.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint and JSON-safe kwargs used to rebuild this head on Ray
            workers.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_fqf_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
