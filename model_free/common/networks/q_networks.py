"""Q-network families for value-based reinforcement learning.

This module includes classic DQN-style heads, recurrent DRQN variants, and
distributional/quantile extensions (C51, QR-DQN, IQN, FQF, Rainbow-adjacent
components). Implementations share common utilities for feature extraction,
dueling decomposition, initialization, and batch-shape normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Type

import math
import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.base_networks import BaseValueNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor, NoisyLinear, NoisyMLPFeaturesExtractor
from rllib.model_free.common.utils.network_utils import DuelingMixin, _ensure_batch, _make_weights_init


# =============================================================================
# Constants
# =============================================================================

PROB_EPS = 1e-6
"""
Small constant used to avoid exact zeros in categorical distributions.

Notes
-----
For distributional RL (C51), probabilities close to 0 can cause numerical issues
(e.g., when taking log or when projecting distributions). Clamping to PROB_EPS
is a common practical stabilization trick.
"""


# =============================================================================
# Standard DQN Q-network
# =============================================================================

class QNetwork(BaseValueNetwork, DuelingMixin):
    """
    Discrete-action Q-network (DQN-style), optionally with dueling decomposition.

    This network maps states to Q-values over discrete actions:
        Q(s) ∈ R^{A}

    If dueling is enabled, Q-values are computed via:
        Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class for trunk (default: ``nn.ReLU``).
    dueling_mode : bool, optional
        If True, use dueling heads (default: False).
    init_type : str, optional
        Weight initializer scheme name passed to `_make_weights_init`
        (default: "orthogonal").
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Returns
    -------
    torch.Tensor
        Q-values of shape (B, action_dim).

    Attributes
    ----------
    action_dim : int
        Number of actions.
    dueling_mode : bool
        Whether dueling decomposition is enabled.
    trunk : MLPFeaturesExtractor
        Shared trunk feature extractor (from BaseValueNetwork).
    trunk_dim : int
        Output dimension of trunk features.
    q_head : nn.Linear
        Q head producing (B, A), present iff dueling_mode=False.
    value_head : nn.Linear
        Value head producing (B, 1), present iff dueling_mode=True.
    adv_head : nn.Linear
        Advantage head producing (B, A), present iff dueling_mode=True.

    Notes
    -----
    Initialization:
    - We intentionally apply initialization AFTER creating heads so that heads
      are included in initialization. This avoids subtle bugs where `super().__init__`
      would initialize only the trunk (or only modules that exist at that time).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        dueling_mode: bool = False,
        feature_extractor: nn.Module | None = None,
        feature_dim: int | None = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        state_dim : Any
            Argument ``state_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        action_dim : Any
            Argument ``action_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        dueling_mode : Any
            Argument ``dueling_mode`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor : Any
            Argument ``feature_extractor`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_trunk : Any
            Argument ``init_trunk`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        self.action_dim = int(action_dim)
        self.dueling_mode = bool(dueling_mode)

        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            feature_extractor=feature_extractor,
            feature_dim=feature_dim,
            init_trunk=init_trunk,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        if self.dueling_mode:
            self.value_head = nn.Linear(self.trunk_dim, 1)             # (B, 1)
            self.adv_head = nn.Linear(self.trunk_dim, self.action_dim) # (B, A)
        else:
            self.q_head = nn.Linear(self.trunk_dim, self.action_dim)   # (B, A)

        # Apply init after heads exist (trunk + heads).
        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if self._external_trunk and (not self._init_trunk):
            if self.dueling_mode:
                self.value_head.apply(init_fn)
                self.adv_head.apply(init_fn)
            else:
                self.q_head.apply(init_fn)
        else:
            self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute Q-values.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Q-values tensor of shape (B, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat)   # (B, 1)
            a = self.adv_head(feat)     # (B, A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat)


class RecurrentQNetwork(nn.Module):
    """GRU-based Q network for discrete actions.

    Parameters
    ----------
    state_dim : int
        Flattened observation dimension.
    action_dim : int
        Number of discrete actions.
    hidden_sizes : Sequence[int], default=(128,)
        MLP sizes applied before the recurrent block.
    rnn_hidden_size : int, default=128
        GRU hidden state size.
    rnn_num_layers : int, default=1
        Number of GRU layers.
    activation_fn : Any, default=nn.ReLU
        Activation module class used in the MLP trunk.
    dueling_mode : bool, default=False
        If True, use dueling value/advantage heads.
    feature_extractor : nn.Module or None, default=None
        Optional feature extractor produced by ``build_feature_extractor``.
    feature_dim : int or None, default=None
        Output feature dimension of the feature extractor.
    init_type : str, default="orthogonal"
        Initialization strategy name.
    gain : float, default=1.0
        Initialization gain.
    bias : float, default=0.0
        Initialization bias constant.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128,),
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        feature_extractor: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.dueling_mode = bool(dueling_mode)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.rnn_num_layers = int(rnn_num_layers)

        self.feature_extractor = feature_extractor
        in_dim = int(feature_dim if feature_extractor is not None and feature_dim is not None else state_dim)

        hidden_sizes_t = tuple(int(h) for h in hidden_sizes)
        layers: list[nn.Module] = []
        prev = in_dim
        act_cls = activation_fn if activation_fn is not None else nn.ReLU
        for h in hidden_sizes_t:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        trunk_dim = prev

        self.rnn = nn.GRU(
            input_size=trunk_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

        if self.dueling_mode:
            self.value_head = nn.Linear(self.rnn_hidden_size, 1)
            self.adv_head = nn.Linear(self.rnn_hidden_size, self.action_dim)
        else:
            self.q_head = nn.Linear(self.rnn_hidden_size, self.action_dim)

        self.apply(_make_weights_init(init_type=init_type, gain=float(gain), bias=float(bias)))

    def _extract(self, obs_flat: th.Tensor) -> th.Tensor:
        """Apply optional feature extractor and trunk MLP to flattened observations."""
        x = obs_flat
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        return self.trunk(x)

    def forward(
        self,
        obs: th.Tensor,
        hidden: Optional[th.Tensor] = None,
        *,
        return_hidden: bool = False,
    ) -> th.Tensor | Tuple[th.Tensor, th.Tensor]:
        """Compute Q-values for one step or a sequence."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        is_step = (obs.dim() == 2)
        if is_step:
            obs = obs.unsqueeze(1)  # (B,1,D)
        elif obs.dim() != 3:
            raise ValueError(f"Expected obs dim 2 or 3, got {tuple(obs.shape)}")

        bsz, tlen = int(obs.shape[0]), int(obs.shape[1])
        flat = obs.reshape(bsz * tlen, *obs.shape[2:])
        feat = self._extract(flat).view(bsz, tlen, -1)

        out, hidden_next = self.rnn(feat, hidden)
        out_flat = out.reshape(bsz * tlen, self.rnn_hidden_size)
        if self.dueling_mode:
            v = self.value_head(out_flat)
            a = self.adv_head(out_flat)
            q_flat = v + (a - a.mean(dim=-1, keepdim=True))
        else:
            q_flat = self.q_head(out_flat)
        q_seq = q_flat.view(bsz, tlen, self.action_dim)
        q_out = q_seq[:, -1, :] if is_step else q_seq

        if return_hidden:
            return q_out, hidden_next
        return q_out


class DoubleQNetwork(nn.Module):
    """
    Twin Q-network wrapper for discrete actions (independent networks).

    This module is commonly used for Double DQN-style target computation
    (or for algorithms that maintain two estimators to reduce overestimation).

    Parameters
    ----------
    *args, **kwargs
        Forwarded to `QNetwork`.

    Attributes
    ----------
    q1 : QNetwork
        First Q network.
    q2 : QNetwork
        Second Q network.

    Returns
    -------
    q1, q2 : torch.Tensor
        Each tensor has shape (B, action_dim).
    """

    def __init__(
        self,
        *args: Any,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        init_trunk: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_trunk : Any
            Argument ``init_trunk`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        *args : Any
            Argument ``*args`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        **kwargs : Any
            Argument ``**kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()
        # Build separate feature extractors for each Q network if provided.
        if feature_extractor_cls is not None:
            obs_dim = int(kwargs.get("state_dim", args[0] if args else 0))
            fe1, fd1 = build_feature_extractor(
                obs_dim=obs_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )
            fe2, fd2 = build_feature_extractor(
                obs_dim=obs_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )
            self.q1 = QNetwork(*args, feature_extractor=fe1, feature_dim=fd1, init_trunk=init_trunk, **kwargs)
            self.q2 = QNetwork(*args, feature_extractor=fe2, feature_dim=fd2, init_trunk=init_trunk, **kwargs)
        else:
            self.q1 = QNetwork(*args, init_trunk=init_trunk, **kwargs)
            self.q2 = QNetwork(*args, init_trunk=init_trunk, **kwargs)

    def forward(self, state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        q1 : torch.Tensor
            Q-values from first network, shape (B, action_dim).
        q2 : torch.Tensor
            Q-values from second network, shape (B, action_dim).
        """
        return self.q1(state), self.q2(state)


# =============================================================================
# Quantile Regression DQN (QR-DQN-style)
# =============================================================================

class FixedQuantileQNetwork(BaseValueNetwork, DuelingMixin):
    """
    Quantile Q-network (QR-DQN style), optionally with dueling decomposition.

    This network outputs quantile estimates for each action:
        Z_θ(s, a) ∈ R^{N}
    returned as a tensor of shape:
        (B, N, A)

    where:
    - B: batch size
    - N: number of quantiles
    - A: number of actions

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    n_quantiles : int, optional
        Number of quantiles (N), default: 200.
    hidden_sizes : tuple[int, ...], optional
        Trunk MLP hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    dueling_mode : bool, optional
        If True, apply dueling decomposition at the quantile level (default: False).
    init_type : str, optional
        Weight initializer (default: "orthogonal").
    gain : float, optional
        Init gain (default: 1.0).
    bias : float, optional
        Bias init constant (default: 0.0).

    Returns
    -------
    torch.Tensor
        Quantile tensor of shape (B, n_quantiles, action_dim).

    Notes
    -----
    Dueling shapes:
    - V(s): (B, N, 1)
    - A(s,a): (B, N, A)
    - Combine using mean over action dimension (mean_dim=-1).

    Initialization:
    - Applied after heads are created (same rationale as `QNetwork`).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 200,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        dueling_mode: bool = False,
        feature_extractor: nn.Module | None = None,
        feature_dim: int | None = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        state_dim : Any
            Argument ``state_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        action_dim : Any
            Argument ``action_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_quantiles : Any
            Argument ``n_quantiles`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        dueling_mode : Any
            Argument ``dueling_mode`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor : Any
            Argument ``feature_extractor`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_trunk : Any
            Argument ``init_trunk`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.dueling_mode = bool(dueling_mode)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got: {self.n_quantiles}")

        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            feature_extractor=feature_extractor,
            feature_dim=feature_dim,
            init_trunk=init_trunk,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        out_dim = self.action_dim * self.n_quantiles

        if self.dueling_mode:
            self.value_head = nn.Linear(self.trunk_dim, self.n_quantiles)  # (B, N)
            self.adv_head = nn.Linear(self.trunk_dim, out_dim)             # (B, A*N)
        else:
            self.q_head = nn.Linear(self.trunk_dim, out_dim)               # (B, A*N)

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if self._external_trunk and (not self._init_trunk):
            if self.dueling_mode:
                self.value_head.apply(init_fn)
                self.adv_head.apply(init_fn)
            else:
                self.q_head.apply(init_fn)
        else:
            self.apply(init_fn)

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute quantile values for each action.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Quantile values of shape (B, n_quantiles, action_dim).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        if self.dueling_mode:
            v = self.value_head(feat).view(-1, self.n_quantiles, 1)               # (B, N, 1)
            a = self.adv_head(feat).view(-1, self.n_quantiles, self.action_dim)   # (B, N, A)
            return self.combine_dueling(v, a, mean_dim=-1)

        return self.q_head(feat).view(-1, self.n_quantiles, self.action_dim)


# =============================================================================
# IQN / FQF quantile-value networks
# =============================================================================

class TauQuantileQNetwork(nn.Module):
    """
    Tau-conditioned quantile value network used by IQN/FQF-style discrete critics.

    Given observations and sampled quantile fractions ``taus`` of shape ``(B, N)``,
    this network returns quantile action values with shape ``(B, N, A)``.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 32,
        n_cos_embeddings: int = 64,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: type[nn.Module] = nn.ReLU,
        dueling_mode: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        obs_dim : Any
            Argument ``obs_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_actions : Any
            Argument ``n_actions`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_quantiles : Any
            Argument ``n_quantiles`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_cos_embeddings : Any
            Argument ``n_cos_embeddings`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        dueling_mode : Any
            Argument ``dueling_mode`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_trunk : Any
            Argument ``init_trunk`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.n_cos_embeddings = int(n_cos_embeddings)
        self.dueling_mode = bool(dueling_mode)

        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be > 0, got {self.n_actions}")
        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got {self.n_quantiles}")
        if self.n_cos_embeddings <= 0:
            raise ValueError(f"n_cos_embeddings must be > 0, got {self.n_cos_embeddings}")

        feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.state_trunk, feat_dim = build_feature_extractor(
            obs_dim=self.obs_dim,
            obs_shape=obs_shape,
            feature_extractor_cls=feature_extractor_cls,
            feature_extractor_kwargs=feature_extractor_kwargs,
        )

        self._external_trunk = self.state_trunk is not None
        self._init_trunk = True if init_trunk is None else bool(init_trunk)

        self.feature_dim = int(feat_dim)
        self.cos_embedding = nn.Linear(self.n_cos_embeddings, self.feature_dim)

        hidden_sizes_t = tuple(int(x) for x in hidden_sizes)
        mlp_layers = []
        in_dim = self.feature_dim
        for hs in hidden_sizes_t:
            mlp_layers += [nn.Linear(in_dim, hs), activation_fn()]
            in_dim = hs
        self.head_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.head_dim = int(in_dim)

        if self.dueling_mode:
            self.value_head = nn.Linear(self.head_dim, 1)
            self.adv_head = nn.Linear(self.head_dim, self.n_actions)
        else:
            self.q_head = nn.Linear(self.head_dim, self.n_actions)

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if self._external_trunk and (not self._init_trunk):
            self.cos_embedding.apply(init_fn)
            self.head_mlp.apply(init_fn)
            if self.dueling_mode:
                self.value_head.apply(init_fn)
                self.adv_head.apply(init_fn)
            else:
                self.q_head.apply(init_fn)
        else:
            self.apply(init_fn)

        idx = th.arange(1, self.n_cos_embeddings + 1, dtype=th.float32)
        self.register_buffer("pi_multiples", idx.view(1, 1, -1) * math.pi)

    def encode_state(self, obs: th.Tensor) -> th.Tensor:
        """Encode observation into latent state features.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        if self.state_trunk is None:
            raise RuntimeError("state_trunk is unexpectedly None")
        return self.state_trunk(obs)

    def forward(self, obs: th.Tensor, taus: th.Tensor) -> th.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        taus : Any
            Argument ``taus`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        state_feat = self.encode_state(obs)
        bsz, n_tau = int(state_feat.shape[0]), int(taus.shape[1])

        cos_in = taus.unsqueeze(-1) * self.pi_multiples.to(device=state_feat.device, dtype=state_feat.dtype)
        tau_feat = th.cos(cos_in)
        tau_feat = th.relu(self.cos_embedding(tau_feat))

        joint = state_feat.unsqueeze(1) * tau_feat
        joint = joint.reshape(bsz * n_tau, self.feature_dim)
        h = self.head_mlp(joint)

        if self.dueling_mode:
            v = self.value_head(h).view(bsz, n_tau, 1)
            a = self.adv_head(h).view(bsz, n_tau, self.n_actions)
            q = v + (a - a.mean(dim=-1, keepdim=True))
        else:
            q = self.q_head(h).view(bsz, n_tau, self.n_actions)

        return q


class FractionProposalNetwork(nn.Module):
    """
    Fraction proposal network for FQF.

    Maps encoded state features ``(B, D)`` to adaptive quantile fractions:
    - taus: ``(B, N+1)`` cumulative fractions in [0,1]
    - tau_hats: ``(B, N)`` midpoint fractions
    - entropy: ``(B,)`` categorical entropy of fraction probabilities
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        n_quantiles: int,
        hidden_size: int = 128,
        activation_fn: type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_quantiles : Any
            Argument ``n_quantiles`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_size : Any
            Argument ``hidden_size`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()
        self.n_quantiles = int(n_quantiles)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be > 0, got: {self.n_quantiles}")

        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_size)),
            activation_fn(),
            nn.Linear(int(hidden_size), self.n_quantiles),
        )

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def forward(self, state_features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        state_features : Any
            Argument ``state_features`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        logits = self.net(state_features)  # (B, N)
        probs = th.softmax(logits, dim=-1).clamp(min=1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        cdf = probs.cumsum(dim=-1)  # (B, N)
        tau_0 = th.zeros((state_features.shape[0], 1), device=state_features.device, dtype=state_features.dtype)
        taus = th.cat([tau_0, cdf], dim=-1)  # (B, N+1)

        tau_hats = 0.5 * (taus[:, :-1] + taus[:, 1:])  # (B, N)
        entropy = -(probs * probs.log()).sum(dim=-1)  # (B,)
        return taus, tau_hats, entropy


# =============================================================================
# Rainbow / C51 with Noisy + Dueling (Distributional)
# =============================================================================

class RainbowQNetwork(nn.Module, DuelingMixin):
    """
    Rainbow-style C51 Q-network (NoisyNet + Dueling + Distributional head).

    This network models the return distribution Z(s, a) as a categorical
    distribution over a fixed support (atoms).

    Outputs
    -------
    - dist(state):
        Probability distribution over atoms for each action:
        shape (B, action_dim, atom_size), sums to 1 over last dimension.
    - forward(state):
        Expected Q-values:
        shape (B, action_dim).
    - reset_noise():
        Resample all NoisyLinear noise buffers (typically called every step).

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Number of discrete actions (A).
    atom_size : int
        Number of atoms (K).
    support : torch.Tensor
        Support values of shape (K,). Will be registered as a buffer.
    hidden_sizes : tuple[int, ...], optional
        Noisy trunk hidden sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class for trunk (default: ``nn.ReLU``).
    init_type : str, optional
        Initializer name for deterministic parts (default: "orthogonal").
    gain : float, optional
        Init gain for deterministic parts (default: 1.0).
    bias : float, optional
        Bias init constant for deterministic parts (default: 0.0).
    noisy_std_init : float, optional
        Initial sigma for NoisyLinear layers (default: 0.5).

    Attributes
    ----------
    action_dim : int
        Number of actions.
    atom_size : int
        Number of atoms.
    support : torch.Tensor
        Registered buffer of shape (K,).
    trunk : NoisyMLPFeaturesExtractor
        Noisy feature extractor.
    value_layer : NoisyLinear
        Value head producing logits for atoms, shape (B, K) before reshaping.
    adv_layer : NoisyLinear
        Advantage head producing logits, shape (B, A*K) before reshaping.

    Notes
    -----
    Dueling combination (logits space):
        logits = V + (A - mean_a A)
    where mean is taken over action dimension (dim=1) for logits shaped (B, A, K).

    Probability stabilization:
    - `softmax(logits)` produces probabilities; we clamp to `PROB_EPS` and renormalize
      to avoid exact zeros.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_size: int,
        support: th.Tensor,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        feature_extractor: nn.Module | None = None,
        feature_dim: int | None = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.5,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        state_dim : Any
            Argument ``state_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        action_dim : Any
            Argument ``action_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        atom_size : Any
            Argument ``atom_size`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        support : Any
            Argument ``support`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor : Any
            Argument ``feature_extractor`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_trunk : Any
            Argument ``init_trunk`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        noisy_std_init : Any
            Argument ``noisy_std_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()

        self.action_dim = int(action_dim)
        self.atom_size = int(atom_size)

        if self.atom_size <= 0:
            raise ValueError(f"atom_size must be > 0, got: {self.atom_size}")
        if support.numel() != self.atom_size:
            raise ValueError(
                f"support must have numel()==atom_size ({self.atom_size}), got: {support.numel()}"
            )

        self._external_trunk = feature_extractor is not None
        self._init_trunk = True if init_trunk is None else bool(init_trunk)

        if feature_extractor is not None:
            self.trunk = feature_extractor
            if feature_dim is not None:
                feat_dim = int(feature_dim)
            else:
                inferred = getattr(self.trunk, "out_dim", None)
                if inferred is None:
                    raise ValueError(
                        "feature_dim must be provided when feature_extractor has no out_dim attribute."
                    )
                feat_dim = int(inferred)
        else:
            self.trunk = NoisyMLPFeaturesExtractor(
                input_dim=int(state_dim),
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                init_type=init_type,
                gain=gain,
                bias=bias,
                noisy_std_init=noisy_std_init,
            )
            feat_dim = int(self.trunk.out_dim)

        # Keep support on the same device and in state_dict.
        self.register_buffer("support", support.view(-1))

        self.value_layer = NoisyLinear(feat_dim, self.atom_size, std_init=noisy_std_init)                  # (B, K)
        self.adv_layer = NoisyLinear(feat_dim, self.action_dim * self.atom_size, std_init=noisy_std_init)  # (B, A*K)

        if self._external_trunk and self._init_trunk:
            init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
            self.trunk.apply(init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has batch dimension and is moved to module device.

        Parameters
        ----------
        x : Any
            Tensor/ndarray/sequence input convertible by `_ensure_batch`.

        Returns
        -------
        torch.Tensor
            Tensor on module device with a batch dimension.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    def dist(self, state: th.Tensor) -> th.Tensor:
        """
        Compute categorical distribution over atoms for each action.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Probability distribution over atoms, shape (B, action_dim, atom_size).
            Each (B, action_dim, :) slice sums to 1 over the last dimension.

        Notes
        -----
        - The dueling combination is performed in logits space.
        - Probabilities are clamped and renormalized for numerical stability.
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)

        adv = self.adv_layer(feat).view(-1, self.action_dim, self.atom_size)  # (B, A, K)
        val = self.value_layer(feat).view(-1, 1, self.atom_size)              # (B, 1, K)

        # Mean over action dimension (dim=1) for logits in (B, A, K).
        logits = self.combine_dueling(val, adv, mean_dim=1)

        prob = th.softmax(logits, dim=-1)
        prob = prob.clamp(min=PROB_EPS)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        return prob

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Compute expected Q-values from the categorical distribution.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Expected Q-values, shape (B, action_dim).

        Notes
        -----
        Expectation is computed as:
            Q(s,a) = sum_k p_k(s,a) * support_k
        """
        prob = self.dist(state)  # (B, A, K)
        return th.sum(prob * self.support.view(1, 1, -1), dim=-1)

    def reset_noise(self) -> None:
        """
        Resample noise in trunk and heads (NoisyNet exploration).

        Notes
        -----
        Typical usage is to call this once per environment step so that the
        policy induced by the network changes over time via parameter noise.
        """
        if hasattr(self.trunk, "reset_noise"):
            self.trunk.reset_noise()
        self.value_layer.reset_noise()
        self.adv_layer.reset_noise()
