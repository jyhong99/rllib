"""Base abstractions and shared building blocks for RL networks.

This module defines reusable base classes for policy, value, and critic
networks. Concrete algorithm-specific modules compose these classes to share
common behavior such as feature extraction wiring, hidden-layer validation,
parameter initialization, and batch-shape handling.

The base classes are intentionally framework-local (PyTorch) and are designed
to be lightweight wrappers around standard ``nn.Module`` patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import torch as th
import torch.nn as nn

from rllib.model_free.common.utils.network_utils import _ensure_batch, _make_weights_init, _validate_hidden_sizes
from rllib.model_free.common.networks.feature_extractors import MLPFeaturesExtractor, build_feature_extractor


class BasePolicyNetwork(nn.Module, ABC):
    """Common trunk + rollout interface for policies."""

    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_sizes: list[int],
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
        obs_dim : Any
            Argument ``obs_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        super().__init__()
        hs = _validate_hidden_sizes(hidden_sizes)

        self.obs_dim = int(obs_dim)
        self.hidden_sizes = list(hs)
        self.activation_fn = activation_fn

        if feature_extractor is None and feature_extractor_cls is not None:
            feature_extractor, feature_dim = build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )

        self._external_trunk = feature_extractor is not None
        if feature_extractor is None:
            self.trunk = MLPFeaturesExtractor(self.obs_dim, self.hidden_sizes, self.activation_fn)
            self.trunk_dim = int(self.trunk.out_dim)
        else:
            self.trunk = feature_extractor
            if feature_dim is None:
                feature_dim = getattr(feature_extractor, "out_dim", None)
            if feature_dim is None:
                raise ValueError("feature_extractor must provide out_dim or feature_dim.")
            self.trunk_dim = int(feature_dim)

        self._init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)

        if init_trunk is None:
            init_trunk = not self._external_trunk
        if bool(init_trunk):
            self.trunk.apply(self._init_fn)

    def _init_module(self, module: nn.Module) -> None:
        """Initialize parameters for a module.

        Parameters
        ----------
        module : Any
            Argument ``module`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        module.apply(self._init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """Ensure batch dimension exists.

        Parameters
        ----------
        x : Any
            Argument ``x`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    @abstractmethod
    @th.no_grad()
    def act(self, obs: th.Tensor, *args: Any, **kwargs: Any) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """Compute an action.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        *args : Any
            Argument ``*args`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        **kwargs : Any
            Argument ``**kwargs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        raise NotImplementedError


class BaseContinuousStochasticPolicy(BasePolicyNetwork, ABC):
    """Abstract base for continuous stochastic policies.

    This class extends :class:`BasePolicyNetwork` with Gaussian parameter heads
    (mean and log-standard-deviation) and exposes a common helper for
    constructing distribution parameters from observations.
    """
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
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
        obs_dim : Any
            Argument ``obs_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        action_dim : Any
            Argument ``action_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        log_std_mode : Any
            Argument ``log_std_mode`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        log_std_init : Any
            Argument ``log_std_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        self.action_dim = int(action_dim)
        super().__init__(
            obs_dim=int(obs_dim),
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

        self.mu = nn.Linear(self.trunk_dim, self.action_dim)
        self._init_module(self.mu)

        self.log_std_mode = str(log_std_mode).lower().strip()
        if self.log_std_mode == "param":
            self.log_std = nn.Parameter(th.ones(self.action_dim) * float(log_std_init))
        elif self.log_std_mode == "layer":
            self.log_std = nn.Linear(self.trunk_dim, self.action_dim)
            self._init_module(self.log_std)
        else:
            raise ValueError(f"Unknown log_std_mode: {log_std_mode!r}")

    def _dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Compute distribution parameters.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)

        mean = self.mu(feat)
        if self.log_std_mode == "param":
            log_std = self.log_std.expand_as(mean)  # type: ignore[union-attr]
        else:
            log_std = self.log_std(feat)  # type: ignore[operator]
        return mean, log_std

    @abstractmethod
    def get_dist(self, obs: th.Tensor):
        """Build and return a distribution object.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        raise NotImplementedError


class BaseDiscreteStochasticPolicy(BasePolicyNetwork, ABC):
    """Abstract base for discrete stochastic policies.

    This class provides a logits head over discrete actions and leaves concrete
    distribution construction to subclasses.
    """
    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
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
        obs_dim : Any
            Argument ``obs_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        n_actions : Any
            Argument ``n_actions`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
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
        self.n_actions = int(n_actions)
        super().__init__(
            obs_dim=int(obs_dim),
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

        self.logits = nn.Linear(self.trunk_dim, self.n_actions)
        self._init_module(self.logits)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        obs = self._ensure_batch(obs)
        feat = self.trunk(obs)
        return self.logits(feat)

    @abstractmethod
    def get_dist(self, obs: th.Tensor):
        """Build and return a distribution object.

        Parameters
        ----------
        obs : Any
            Argument ``obs`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        raise NotImplementedError


class BaseValueNetwork(nn.Module, ABC):
    """Abstract state-value style network with optional shared encoder.

    Subclasses implement task-specific output heads while this base handles
    feature-extractor wiring, hidden-size validation, and initialization policy.
    """
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_sizes: Tuple[int, ...],
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
        apply_init: bool = True,
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
        apply_init : Any
            Argument ``apply_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        if feature_extractor is None and feature_extractor_cls is not None:
            feature_extractor, feature_dim = build_feature_extractor(
                obs_dim=self.state_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )

        self._external_trunk = feature_extractor is not None
        if feature_extractor is None:
            self.trunk = MLPFeaturesExtractor(self.state_dim, list(self.hidden_sizes), self.activation_fn)
            self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))
        else:
            self.trunk = feature_extractor
            if feature_dim is None:
                feature_dim = getattr(feature_extractor, "out_dim", None)
            if feature_dim is None:
                raise ValueError("feature_extractor must provide out_dim or feature_dim.")
            self.trunk_dim = int(feature_dim)

        self._init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        if init_trunk is None:
            init_trunk = (not self._external_trunk) and bool(apply_init)
        self._init_trunk = bool(init_trunk)
        if self._init_trunk:
            self.apply(self._init_fn)

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """Ensure batch dimension exists.

        Parameters
        ----------
        x : Any
            Argument ``x`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)


class BaseCriticNet(nn.Module, ABC):
    """Base class for critic-family modules with shared init utilities."""
    def __init__(self, *, init_type: str, gain: float, bias: float) -> None:
        """Initialize this module.

        Parameters
        ----------
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
        self._init_type = str(init_type)
        self._gain = float(gain)
        self._bias = float(bias)
        self._external_trunk = False
        self._init_trunk = True

    def _ensure_batch(self, x: Any) -> th.Tensor:
        """Ensure batch dimension exists.

        Parameters
        ----------
        x : Any
            Argument ``x`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        device = next(self.parameters()).device
        return _ensure_batch(x, device=device)

    def _finalize_init(self) -> None:
        """Apply initialization policy to critic submodules.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        init_fn = _make_weights_init(
            init_type=self._init_type,
            gain=self._gain,
            bias=self._bias,
        )
        if self._external_trunk and (not self._init_trunk):
            for name, module in self.named_modules():
                if name.startswith("trunk"):
                    continue
                if isinstance(module, nn.Linear):
                    init_fn(module)
            return
        self.apply(init_fn)


class BaseStateCritic(BaseCriticNet):
    """Base critic that consumes only state/observation features."""
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_sizes: Tuple[int, ...],
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
        super().__init__(init_type=init_type, gain=gain, bias=bias)
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn

        if feature_extractor is None and feature_extractor_cls is not None:
            feature_extractor, feature_dim = build_feature_extractor(
                obs_dim=self.state_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )

        self._external_trunk = feature_extractor is not None
        if feature_extractor is None:
            self.trunk = MLPFeaturesExtractor(self.state_dim, list(self.hidden_sizes), activation_fn)
            self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))
        else:
            self.trunk = feature_extractor
            if feature_dim is None:
                feature_dim = getattr(feature_extractor, "out_dim", None)
            if feature_dim is None:
                raise ValueError("feature_extractor must provide out_dim or feature_dim.")
            self.trunk_dim = int(feature_dim)

        if init_trunk is None:
            init_trunk = not self._external_trunk
        self._init_trunk = bool(init_trunk)


class BaseStateActionCritic(BaseCriticNet):
    """Base critic that consumes concatenated state and action features."""
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...],
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
        super().__init__(init_type=init_type, gain=gain, bias=bias)
        hs = _validate_hidden_sizes(hidden_sizes)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(hs)
        self.activation_fn = activation_fn
        self.input_dim = self.state_dim + self.action_dim

        if feature_extractor is None and feature_extractor_cls is not None:
            feature_extractor, feature_dim = build_feature_extractor(
                obs_dim=self.input_dim,
                obs_shape=obs_shape,
                feature_extractor_cls=feature_extractor_cls,
                feature_extractor_kwargs=feature_extractor_kwargs,
            )

        self._external_trunk = feature_extractor is not None
        if feature_extractor is None:
            self.trunk = MLPFeaturesExtractor(self.input_dim, list(self.hidden_sizes), activation_fn)
            self.trunk_dim = int(getattr(self.trunk, "out_dim", self.hidden_sizes[-1]))
        else:
            self.trunk = feature_extractor
            if feature_dim is None:
                feature_dim = getattr(feature_extractor, "out_dim", None)
            if feature_dim is None:
                raise ValueError("feature_extractor must provide out_dim or feature_dim.")
            self.trunk_dim = int(feature_dim)

        if init_trunk is None:
            init_trunk = not self._external_trunk
        self._init_trunk = bool(init_trunk)
