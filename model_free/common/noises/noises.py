"""Action-independent exploration-noise implementations.

This module provides reusable noise processes that can be sampled without
conditioning on the current action. Implementations include i.i.d. Gaussian,
stateful Ornstein-Uhlenbeck, and i.i.d. Uniform noise.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import math
import torch as th

from rllib.model_free.common.noises.base_noise import BaseNoise
from rllib.model_free.common.utils.noise_utils import _normalize_size


# =============================================================================
# Action-independent noise processes
# =============================================================================

def _validate_floating_dtype(dtype: th.dtype) -> None:
    """Validate that a dtype supports random floating-point sampling.

    Parameters
    ----------
    dtype : torch.dtype
        Dtype requested by a noise process.

    Raises
    ------
    TypeError
        If ``dtype`` is not floating-point.
    """
    if not th.is_floating_point(th.empty((), dtype=dtype)):
        raise TypeError(f"dtype must be floating-point, got {dtype}.")

class GaussianNoise(BaseNoise):
    """
    Stateless i.i.d. Gaussian noise.

    Each call produces independent samples:

        noise ~ Normal(mu, sigma^2)

    Parameters
    ----------
    size : int or tuple[int, ...]
        Output shape of the noise tensor. This typically matches an action shape
        such as ``(act_dim,)`` or ``(batch, act_dim)`` depending on usage.
        The value is normalized via :func:`~_normalize_size`.
    mu : float, optional
        Mean of the Gaussian distribution (default: 0.0).
    sigma : float, optional
        Standard deviation of the Gaussian distribution (must be >= 0)
        (default: 0.2).
    device : str or torch.device, optional
        Device on which samples are generated (default: "cpu").
    dtype : torch.dtype, optional
        Data type of generated samples (default: ``torch.float32``).

    Attributes
    ----------
    size : tuple[int, ...]
        Normalized output shape.
    mu : float
        Mean parameter.
    sigma : float
        Standard deviation parameter.
    device : str or torch.device
        Target device for samples.
    dtype : torch.dtype
        Target dtype for samples.

    Returns
    -------
    torch.Tensor
        Noise tensor of shape ``size`` on `device` with dtype `dtype`.

    Notes
    -----
    - Stateless: `reset()` is a no-op (inherited from :class:`~BaseNoise`).
    - If ``sigma == 0``, `sample()` returns a constant tensor filled with `mu`.
      This is often useful for deterministic debugging.

    Examples
    --------
    >>> n = GaussianNoise(size=(3,), mu=0.0, sigma=0.1)
    >>> x = n.sample()  # shape (3,)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        mu: float = 0.0,
        sigma: float = 0.2,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        """
        Initialize a stateless Gaussian noise process.

        Parameters
        ----------
        size : int or tuple[int, ...]
            Output sample shape. A scalar integer is normalized to a 1D tuple.
        mu : float, default=0.0
            Mean of the Gaussian distribution.
        sigma : float, default=0.2
            Standard deviation (must be non-negative).
        device : str or torch.device, default="cpu"
            Device where samples are generated.
        dtype : torch.dtype, default=torch.float32
            Floating dtype of generated samples.

        Raises
        ------
        ValueError
            If ``sigma < 0``.
        TypeError
            If ``dtype`` is not floating-point.
        """
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        _validate_floating_dtype(dtype)

        self.size = _normalize_size(size)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.device = device
        self.dtype = dtype

    def sample(self) -> th.Tensor:
        """
        Draw i.i.d. Gaussian noise.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape ``self.size``.

        Notes
        -----
        If ``self.sigma == 0``, this returns a constant tensor filled with `mu`.
        """
        if self.sigma == 0.0:
            return th.full(self.size, self.mu, device=self.device, dtype=self.dtype)
        return self.mu + self.sigma * th.randn(self.size, device=self.device, dtype=self.dtype)


class OrnsteinUhlenbeckNoise(BaseNoise):
    """
    Ornstein–Uhlenbeck (OU) temporally correlated noise (stateful).

    OU noise introduces temporal correlation, often used in DDPG-style exploration.
    Discrete-time Euler-Maruyama approximation:

        x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0, 1)

    where `x_t` is the internal state (and is typically returned as the noise).

    Parameters
    ----------
    size : int or tuple[int, ...]
        Noise tensor shape. Normalized via :func:`~_normalize_size`.
    mu : float, optional
        Long-run mean (default: 0.0).
    theta : float, optional
        Mean-reversion speed (must be >= 0) (default: 0.15).
    sigma : float, optional
        Volatility / diffusion scale (must be >= 0) (default: 0.2).
    dt : float, optional
        Discrete time step (must be > 0) (default: 1e-2).
    x0 : Optional[float], optional
        Initial value for the internal state. If None, initializes to `mu`
        (default: None).
    return_copy : bool, optional
        If True, return a cloned tensor from `sample()` to prevent external
        in-place modification from corrupting internal state (default: True).
    device : str or torch.device, optional
        Device for internal state and returned samples (default: "cpu").
    dtype : torch.dtype, optional
        Dtype for internal state and returned samples (default: ``torch.float32``).

    Attributes
    ----------
    size : tuple[int, ...]
        Normalized shape of the noise/state.
    mu : float
        Long-run mean parameter.
    theta : float
        Mean-reversion speed.
    sigma : float
        Volatility parameter.
    dt : float
        Time step.
    return_copy : bool
        Whether `sample()` returns a clone.
    state : torch.Tensor
        Internal OU state tensor of shape ``size``.
    device : str or torch.device
        Device used for state and samples.
    dtype : torch.dtype
        Dtype used for state and samples.

    Returns
    -------
    torch.Tensor
        OU noise sample of shape ``size``.

    Notes
    -----
    - Stateful: call :meth:`reset` at episode boundaries.
    - `sample()` updates internal `state` in-place.
    - Degenerate cases:
        - If `sigma == 0` and `theta == 0`, the state never changes after reset.
        - If `sigma == 0` but `theta > 0`, the state deterministically relaxes to `mu`.

    Examples
    --------
    >>> ou = OrnsteinUhlenbeckNoise(size=3, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2)
    >>> ou.reset()
    >>> x = ou.sample()
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[float] = None,
        *,
        return_copy: bool = True,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        """
        Initialize a stateful Ornstein-Uhlenbeck noise process.

        Parameters
        ----------
        size : int or tuple[int, ...]
            State/sample shape.
        mu : float, default=0.0
            Long-run mean target.
        theta : float, default=0.15
            Mean-reversion rate (must be non-negative).
        sigma : float, default=0.2
            Diffusion scale (must be non-negative).
        dt : float, default=1e-2
            Integration time step (must be positive).
        x0 : float, optional
            Initial state value. If omitted, defaults to ``mu``.
        return_copy : bool, default=True
            Whether :meth:`sample` returns a cloned tensor instead of the
            internal state reference.
        device : str or torch.device, default="cpu"
            Device for internal state and random draws.
        dtype : torch.dtype, default=torch.float32
            Floating dtype of state and samples.

        Raises
        ------
        ValueError
            If ``theta < 0``, ``sigma < 0``, or ``dt <= 0``.
        TypeError
            If ``dtype`` is not floating-point.
        """
        if theta < 0.0:
            raise ValueError(f"theta must be >= 0, got {theta}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        _validate_floating_dtype(dtype)

        self.size = _normalize_size(size)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self._sigma_sqrt_dt = self.sigma * math.sqrt(self.dt)
        self.return_copy = bool(return_copy)
        self.device = device
        self.dtype = dtype

        self._x0 = self.mu if x0 is None else float(x0)

        # Ensure state exists immediately.
        self.state: th.Tensor
        self.reset()

    def sample(self) -> th.Tensor:
        """
        Draw one OU noise sample and update internal state.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape ``self.size``.

        Notes
        -----
        - The internal `state` is updated in-place.
        - If `return_copy=True`, returns a clone to avoid aliasing and accidental
          external in-place modification of `self.state`.
        """
        if self.sigma == 0.0 and self.theta == 0.0:
            # Fully degenerate: constant state.
            return self.state.clone() if self.return_copy else self.state

        eps = th.randn(self.size, device=self.device, dtype=self.dtype)
        dx = (self.theta * (self.mu - self.state) * self.dt) + (self._sigma_sqrt_dt * eps)
        self.state.add_(dx)

        return self.state.clone() if self.return_copy else self.state

    def reset(self) -> None:
        """
        Reset internal OU state to the initial value.

        Notes
        -----
        The state is set to a constant tensor filled with `x0` (or `mu` if x0 is None).
        """
        self.state = th.full(self.size, self._x0, device=self.device, dtype=self.dtype)


class UniformNoise(BaseNoise):
    """
    Stateless i.i.d. uniform noise.

    Each call produces independent samples:

        noise ~ Uniform(low, high)

    Parameters
    ----------
    size : int or tuple[int, ...]
        Output shape of the noise tensor. Normalized via :func:`~_normalize_size`.
    low : float, optional
        Lower bound (default: -1.0).
    high : float, optional
        Upper bound (must satisfy ``high > low``) (default: 1.0).
    device : str or torch.device, optional
        Device on which samples are generated (default: "cpu").
    dtype : torch.dtype, optional
        Data type of generated samples (default: ``torch.float32``).

    Attributes
    ----------
    size : tuple[int, ...]
        Normalized output shape.
    low : float
        Lower bound.
    high : float
        Upper bound.
    device : str or torch.device
        Target device for samples.
    dtype : torch.dtype
        Target dtype for samples.

    Returns
    -------
    torch.Tensor
        Noise tensor of shape ``size`` on `device` with dtype `dtype`.

    Notes
    -----
    - Stateless: `reset()` is a no-op.
    - Mainly useful for debugging/sanity checks or for very simple exploration baselines.

    Examples
    --------
    >>> n = UniformNoise(size=(2,), low=-0.5, high=0.5)
    >>> x = n.sample()
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        low: float = -1.0,
        high: float = 1.0,
        device: Union[str, th.device] = "cpu",
        dtype: th.dtype = th.float32,
    ) -> None:
        """
        Initialize a stateless uniform noise process.

        Parameters
        ----------
        size : int or tuple[int, ...]
            Output sample shape.
        low : float, default=-1.0
            Inclusive lower bound.
        high : float, default=1.0
            Exclusive upper bound (must satisfy ``high > low``).
        device : str or torch.device, default="cpu"
            Device where samples are generated.
        dtype : torch.dtype, default=torch.float32
            Floating dtype of generated samples.

        Raises
        ------
        ValueError
            If ``high <= low``.
        TypeError
            If ``dtype`` is not floating-point.
        """
        if high <= low:
            raise ValueError(f"high must be > low, got low={low}, high={high}")
        _validate_floating_dtype(dtype)

        self.size = _normalize_size(size)
        self.low = float(low)
        self.high = float(high)
        self.device = device
        self.dtype = dtype

    def sample(self) -> th.Tensor:
        """
        Draw i.i.d. uniform noise.

        Returns
        -------
        noise : torch.Tensor
            Noise tensor of shape ``self.size``.
        """
        return th.empty(self.size, device=self.device, dtype=self.dtype).uniform_(self.low, self.high)
