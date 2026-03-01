"""DRQN Head.

This module defines the recurrent Q-network head used by DRQN and its Ray
worker reconstruction helper.

It provides:

- :func:`build_drqn_head_worker_policy` for CPU-only worker reconstruction.
- :class:`DRQNHead` with online/target recurrent Q-networks and hidden-state
  management for inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.q_networks import RecurrentQNetwork
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)


def build_drqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """Build a CPU DRQN head for Ray rollout workers.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe constructor kwargs for :class:`DRQNHead`.
        The payload is expected to come from
        :meth:`DRQNHead._export_kwargs_json_safe`.

    Returns
    -------
    nn.Module
        Reconstructed head on CPU with training mode disabled where supported.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))
    head = DRQNHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class DRQNHead(QLearningHead):
    """DRQN head with online/target recurrent Q-networks.

    Notes
    -----
    - Uses :class:`RecurrentQNetwork` for both online and target critics.
    - Maintains a cached hidden state for inference-time action selection.
    - Target network is synchronized once at init and then frozen.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128,),
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize recurrent online and target Q-networks.

        Parameters
        ----------
        obs_dim : int
            Observation vector dimension.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : Sequence[int], default=(128,)
            MLP trunk hidden widths used before recurrent layers.
        rnn_hidden_size : int, default=128
            Hidden size of recurrent layers.
        rnn_num_layers : int, default=1
            Number of recurrent layers.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function class for MLP trunk.
        dueling_mode : bool, default=False
            Whether to enable dueling output heads.
        obs_shape : tuple[int, ...] | None, default=None
            Optional raw observation shape for feature extractor build.
        feature_extractor_cls : type[nn.Module] | None, default=None
            Optional feature extractor class.
        feature_extractor_kwargs : dict[str, Any] | None, default=None
            Optional feature extractor kwargs.
        init_type : str, default="orthogonal"
            Initialization scheme identifier.
        gain : float, default=1.0
            Initialization gain.
        bias : float, default=0.0
            Initialization bias value.
        device : str | torch.device
            Module device.

        Raises
        ------
        ValueError
            If action-space or recurrent-network dimensions are invalid.
        """
        super().__init__(device=device)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.rnn_num_layers = int(rnn_num_layers)
        self.activation_fn = activation_fn
        self.dueling_mode = bool(dueling_mode)
        self.obs_shape = obs_shape
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got: {self.n_actions}")
        if self.rnn_hidden_size <= 0:
            raise ValueError(f"rnn_hidden_size must be > 0, got: {self.rnn_hidden_size}")
        if self.rnn_num_layers <= 0:
            raise ValueError(f"rnn_num_layers must be > 0, got: {self.rnn_num_layers}")
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must contain positive integers, got: {self.hidden_sizes}")

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build one feature-extractor pair for one Q-network branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        net_kwargs: Dict[str, Any] = {
            "state_dim": self.obs_dim,
            "action_dim": self.n_actions,
            "hidden_sizes": self.hidden_sizes,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "activation_fn": self.activation_fn,
            "dueling_mode": self.dueling_mode,
            "init_type": self.init_type,
            "gain": self.gain,
            "bias": self.bias,
        }

        def _build_q_branch() -> RecurrentQNetwork:
            """Build one recurrent Q branch with its own feature extractor."""
            fe, fd = _build_fe()
            return RecurrentQNetwork(
                feature_extractor=fe,
                feature_dim=fd,
                **net_kwargs,
            ).to(self.device)

        self.q = _build_q_branch()
        self.q_target = _build_q_branch()

        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)
        self._hidden: Optional[th.Tensor] = None

    def reset_hidden(self) -> None:
        """Reset the cached recurrent hidden state.

        Notes
        -----
        This is typically called:
        - at episode boundaries,
        - when switching between train/eval mode, and
        - after checkpoint loads.
        """
        self._hidden = None

    def reset_exploration_noise(self) -> None:
        """Alias required by the generic off-policy driver.

        Notes
        -----
        DRQN does not use parameter-space noise. The off-policy driver calls
        this method at episode boundaries, so we map it to hidden-state reset.
        """
        self.reset_hidden()

    @th.no_grad()
    def act(
        self,
        obs: Any,
        *,
        epsilon: float = 0.0,
        deterministic: bool = True,
    ) -> th.Tensor:
        """Select actions with recurrent hidden-state carry and epsilon-greedy policy.

        Parameters
        ----------
        obs : Any
            Observation batch or single observation compatible with
            :meth:`_to_tensor_batched`.
        epsilon : float, default=0.0
            Probability of selecting a random action when
            ``deterministic=False``.
        deterministic : bool, default=True
            If ``True``, always returns greedy actions and ignores ``epsilon``.

        Returns
        -------
        torch.Tensor
            Chosen action indices with shape ``(B,)`` and dtype ``torch.long``.
        """
        s = self._to_tensor_batched(obs)
        bsz = int(s.shape[0])
        if self._hidden is not None and int(self._hidden.shape[1]) != bsz:
            self._hidden = None

        q, hidden_next = self.q(s, hidden=self._hidden, return_hidden=True)
        self._hidden = hidden_next
        greedy = th.argmax(q, dim=-1)

        if deterministic or float(epsilon) <= 0.0:
            return greedy.long()

        rand = th.randint(0, int(self.n_actions), (bsz,), device=self.device)
        mask = th.rand(bsz, device=self.device) < float(epsilon)
        return th.where(mask, rand, greedy).long()

    def set_training(self, training: bool) -> None:
        """Set module train/eval mode and clear hidden state.

        Parameters
        ----------
        training : bool
            Desired mode. ``True`` enables training mode, ``False`` enables
            evaluation mode.
        """
        super().set_training(training)
        self.reset_hidden()

    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """Export constructor kwargs in JSON-safe format.

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
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "rnn_hidden_size": int(self.rnn_hidden_size),
            "rnn_num_layers": int(self.rnn_num_layers),
            "activation_fn": self._activation_to_name(self.activation_fn),
            "dueling_mode": bool(self.dueling_mode),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": fe_name,
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """Save head checkpoint with online/target recurrent networks.

        Parameters
        ----------
        path : str
            Output checkpoint path. ``.pt`` is appended if missing.
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
        """Load head checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a checkpoint produced by :meth:`save`.

        Raises
        ------
        ValueError
            If checkpoint format is not recognized.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.q.load_state_dict(ckpt["q"])
        if "q_target" in ckpt and ckpt["q_target"] is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
        else:
            self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)
        self.reset_hidden()

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """Return Ray worker reconstruction specification.

        Returns
        -------
        PolicyFactorySpec
            Entrypoint and JSON-safe kwargs for worker-side policy rebuild.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_drqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
