"""State-value and critic network implementations.

This module implements scalar value estimators and state-action critics used by
on-policy and off-policy algorithms, including ensembles and quantile variants.
Classes reuse common feature extractors and initialization helpers to keep
architectures consistent across algorithm baselines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.base_networks import BaseStateCritic, BaseStateActionCritic
from rllib.model_free.common.networks.feature_extractors import MLPFeaturesExtractor, build_feature_extractor
from rllib.model_free.common.utils.network_utils import _ensure_batch, _make_weights_init, _validate_hidden_sizes


def _ensure_same_batch_size(state: th.Tensor, action: th.Tensor) -> None:
    """Validate that state/action tensors share the same batch size.

    Parameters
    ----------
    state : torch.Tensor
        Batched state tensor.
    action : torch.Tensor
        Batched action tensor.

    Raises
    ------
    ValueError
        If batch sizes differ.
    """
    if state.shape[0] != action.shape[0]:
        raise ValueError(
            "state and action batch sizes must match, "
            f"got {state.shape[0]} and {action.shape[0]}"
        )


# =============================================================================
# V(s)
# =============================================================================

class StateValueNetwork(BaseStateCritic):
    """
    State-value function approximator V(s).

    This network implements a standard value function:
        V(s) = head(trunk(s))

    where `trunk` is an MLP feature extractor and `head` maps features to a scalar.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the trunk MLP (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class inserted between Linear layers (default: ``nn.ReLU``).
    init_type : str, optional
        Weight initialization scheme name (default: ``"orthogonal"``).
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    state_dim : int
        State dimension.
    hidden_sizes : tuple[int, ...]
        Hidden sizes for the trunk.
    trunk : MLPFeaturesExtractor
        Feature extractor mapping (B, state_dim) -> (B, trunk_dim).
    trunk_dim : int
        Feature dimension produced by the trunk.
    head : nn.Linear
        Linear head mapping (B, trunk_dim) -> (B, 1).

    Returns
    -------
    torch.Tensor
        Value estimates V(s), shape (B, 1).

    Notes
    -----
    Initialization:
    - `BaseStateCritic` (via `BaseCriticNet`) stores init hyperparameters.
    - We call `_finalize_init()` after building `head` so both trunk and head
      are initialized consistently.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        feature_extractor: nn.Module | None = None,
        feature_dim: int | None = None,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
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
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor : Any
            Argument ``feature_extractor`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        super().__init__(
            state_dim=int(state_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            feature_extractor=feature_extractor,
            feature_dim=feature_dim,
            feature_extractor_cls=feature_extractor_cls,
            feature_extractor_kwargs=feature_extractor_kwargs,
            obs_shape=obs_shape,
            init_trunk=init_trunk,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.head = nn.Linear(self.trunk_dim, 1)

        # Finalize initialization after all modules exist.
        self._finalize_init()

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).

        Returns
        -------
        torch.Tensor
            Value estimates of shape (B, 1).
        """
        state = self._ensure_batch(state)
        feat = self.trunk(state)
        return self.head(feat)


# =============================================================================
# Q(s,a)
# =============================================================================

class StateActionValueNetwork(BaseStateActionCritic):
    """
    State-action value function approximator Q(s, a).

    This network implements a standard critic:
        Q(s, a) = head(trunk([s, a]))

    where [s, a] denotes concatenation along the last dimension.

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes of the trunk MLP (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    init_type : str, optional
        Weight initialization scheme name (default: ``"orthogonal"``).
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    input_dim : int
        Concatenated input dimension = state_dim + action_dim.
    trunk : MLPFeaturesExtractor
        Feature extractor mapping (B, input_dim) -> (B, trunk_dim).
    head : nn.Linear
        Linear head mapping (B, trunk_dim) -> (B, 1).

    Returns
    -------
    torch.Tensor
        Q-value estimates Q(s, a), shape (B, 1).

    Notes
    -----
    - `state` and `action` must have the same batch size B after `_ensure_batch`.
      If you ever see shape bugs here, add an assertion:
          assert state.size(0) == action.size(0)
    - Initialization is finalized after heads exist via `_finalize_init()`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        feature_extractor: nn.Module | None = None,
        feature_dim: int | None = None,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
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
        feature_extractor : Any
            Argument ``feature_extractor`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_dim : Any
            Argument ``feature_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        super().__init__(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            feature_extractor=feature_extractor,
            feature_dim=feature_dim,
            feature_extractor_cls=feature_extractor_cls,
            feature_extractor_kwargs=feature_extractor_kwargs,
            obs_shape=obs_shape,
            init_trunk=init_trunk,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.head = nn.Linear(self.trunk_dim, 1)

        # Finalize initialization after all modules exist.
        self._finalize_init()

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).
        action : torch.Tensor
            Action tensor of shape (action_dim,) or (B, action_dim).

        Returns
        -------
        torch.Tensor
            Q-value estimates of shape (B, 1).
        """
        state = self._ensure_batch(state)
        action = self._ensure_batch(action)
        _ensure_same_batch_size(state, action)

        x = th.cat([state, action], dim=-1)
        feat = self.trunk(x)
        return self.head(feat)


class DoubleStateActionValueNetwork(nn.Module):
    """
    Twin state-action critics: Q1(s,a), Q2(s,a).

    This wrapper is commonly used in TD3/SAC-style algorithms to reduce
    overestimation bias via clipped double-Q targets.

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    hidden_sizes : tuple[int, ...], optional
        Hidden sizes for each critic trunk (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    init_type : str, optional
        Weight initialization scheme name (default: ``"orthogonal"``).
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    q1 : StateActionValueNetwork
        First critic.
    q2 : StateActionValueNetwork
        Second critic.

    Returns
    -------
    q1, q2 : torch.Tensor
        Each tensor is shape (B, 1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
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
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        if feature_extractor_cls is not None:
            input_dim = int(state_dim) + int(action_dim)
            fe1, fd1 = build_feature_extractor(
                obs_dim=int(state_dim),
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
                input_dim_override=input_dim,
            )
            fe2, fd2 = build_feature_extractor(
                obs_dim=int(state_dim),
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
                input_dim_override=input_dim,
            )
            self.q1 = StateActionValueNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                feature_extractor=fe1,
                feature_dim=fd1,
                init_trunk=init_trunk,
                init_type=init_type,
                gain=gain,
                bias=bias,
            )
            self.q2 = StateActionValueNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                feature_extractor=fe2,
                feature_dim=fd2,
                init_trunk=init_trunk,
                init_type=init_type,
                gain=gain,
                bias=bias,
            )
        else:
            self.q1 = StateActionValueNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                init_trunk=init_trunk,
                init_type=init_type,
                gain=gain,
                bias=bias,
            )
            self.q2 = StateActionValueNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                activation_fn=activation_fn,
                init_trunk=init_trunk,
                init_type=init_type,
                gain=gain,
                bias=bias,
            )

    def forward(self, state: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).
        action : torch.Tensor
            Action tensor of shape (action_dim,) or (B, action_dim).

        Returns
        -------
        q1 : torch.Tensor
            Q1(s,a), shape (B, 1).
        q2 : torch.Tensor
            Q2(s,a), shape (B, 1).
        """
        return self.q1(state, action), self.q2(state, action)


# =============================================================================
# Quantile critic ensemble (TQC/QR variants)
# =============================================================================

class QuantileStateActionValueNetwork(nn.Module):
    """
    Quantile critic ensemble with independent trunks and heads.

    This module outputs quantile values per critic network:
        output shape = (B, n_nets, n_quantiles)

    This is useful for:
    - TQC (Truncated Quantile Critics)
    - QR-style critic ensembles
    - any distributional continuous-control variant

    Parameters
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    n_quantiles : int, optional
        Number of quantiles produced per critic (default: 25).
    n_nets : int, optional
        Number of critics in the ensemble (default: 2).
    hidden_sizes : tuple[int, ...], optional
        Trunk hidden sizes for each critic (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class used in each trunk (default: ``nn.ReLU``).
    init_type : str, optional
        Weight initialization scheme name (default: ``"orthogonal"``).
    gain : float, optional
        Initialization gain (default: 1.0).
    bias : float, optional
        Bias initialization constant (default: 0.0).

    Attributes
    ----------
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    input_dim : int
        Concatenated input dimension = state_dim + action_dim.
    n_quantiles : int
        Number of quantiles per critic.
    n_nets : int
        Number of critics.
    trunks : nn.ModuleList
        List of feature extractors; each maps (B, input_dim) -> (B, trunk_dim).
    heads : nn.ModuleList
        List of linear heads; each maps (B, trunk_dim) -> (B, n_quantiles).

    Returns
    -------
    torch.Tensor
        Quantile tensor of shape (B, n_nets, n_quantiles).

    Notes
    -----
    - This class does not inherit `BaseStateActionCritic` because that base assumes
      a *single* trunk, while here we explicitly maintain multiple independent trunks.
    - `_ensure_batch` moves inputs to the module device and ensures a batch dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 25,
        n_nets: int = 2,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: Type[nn.Module] = nn.ReLU,
        feature_extractor_cls: Optional[Type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
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
        n_nets : Any
            Argument ``n_nets`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_cls : Any
            Argument ``feature_extractor_cls`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        feature_extractor_kwargs : Any
            Argument ``feature_extractor_kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        obs_shape : Any
            Argument ``obs_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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

        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.state_dim + self.action_dim

        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be positive, got: {self.n_quantiles}")
        if self.n_nets <= 0:
            raise ValueError(f"n_nets must be positive, got: {self.n_nets}")

        if feature_extractor_cls is None:
            self.trunks = nn.ModuleList(
                [MLPFeaturesExtractor(self.input_dim, list(hs), activation_fn) for _ in range(self.n_nets)]
            )
            trunk_dim = int(getattr(self.trunks[0], "out_dim", hs[-1]))
        else:
            trunks: list[nn.Module] = []
            trunk_dim = None
            for _ in range(self.n_nets):
                fe, fd = build_feature_extractor(
                    obs_dim=int(self.state_dim),
                    obs_shape=obs_shape,
                    feature_extractor_cls=feature_extractor_cls,
                    feature_extractor_kwargs=feature_extractor_kwargs,
                    input_dim_override=self.input_dim,
                )
                if fe is None or fd is None:
                    raise ValueError("feature_extractor_cls produced no trunk for quantile critics.")
                trunks.append(fe)
                if trunk_dim is None:
                    trunk_dim = int(fd)
            self.trunks = nn.ModuleList(trunks)
            trunk_dim = int(trunk_dim)

        self.heads = nn.ModuleList([nn.Linear(trunk_dim, self.n_quantiles) for _ in range(self.n_nets)])

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.apply(init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """
        Ensure input has batch dimension and is placed on module device.

        Parameters
        ----------
        x : Any
            Input convertible by `_ensure_batch` (Tensor / ndarray / sequence).

        Returns
        -------
        torch.Tensor
            Tensor on this module's device with shape (B, D).
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Compute quantile values for each critic in the ensemble.

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (state_dim,) or (B, state_dim).
        action : torch.Tensor
            Action tensor of shape (action_dim,) or (B, action_dim).

        Returns
        -------
        torch.Tensor
            Quantile values of shape (B, n_nets, n_quantiles).

        Notes
        -----
        - All critics share the same input x = concat(state, action),
          but have independent parameters.
        - If you want vectorized evaluation (no Python loop), you could stack
          parameters or use vmap; however, explicit loops are often fine given
          small n_nets (e.g., 2~5).
        """
        state = self._ensure_batch(state)
        action = self._ensure_batch(action)
        _ensure_same_batch_size(state, action)

        x = th.cat([state, action], dim=-1)

        qs: list[th.Tensor] = []
        for trunk, head in zip(self.trunks, self.heads):
            feat = trunk(x)         # (B, trunk_dim)
            qs.append(head(feat))   # (B, n_quantiles)

        return th.stack(qs, dim=1)  # (B, n_nets, n_quantiles)
