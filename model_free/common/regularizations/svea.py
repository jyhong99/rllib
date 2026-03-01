"""SVEA regularization and augmentation helpers.

This module provides augmentation primitives and a loss-mixing utility commonly
used in SVEA-style pixel-based RL pipelines:

- random convolution augmentation (domain randomization in image space),
- paired augmentation helpers for ``(obs, next_obs)`` batches,
- weighted clean/augmented loss composition.

The implementations are designed to be composable with replay-based off-policy
algorithms where observation tensors are shaped ``(B, C, H, W)``.
"""

from __future__ import annotations

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _check_image_batch(x: th.Tensor) -> None:
    """Validate image batch tensor shape.

    Parameters
    ----------
    x : torch.Tensor
        Candidate image batch tensor.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.Tensor``.
    ValueError
        If ``x`` does not have shape rank 4.
    """
    if not isinstance(x, th.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"expected shape (B,C,H,W), got shape={tuple(x.shape)}")


class RandomConvAug(nn.Module):
    """Random convolution augmentation used by SVEA.

    Parameters
    ----------
    kernel_size : int, default=3
        Convolution kernel size.
    pad_mode : str, default="replicate"
        Padding mode used before applying random convolution.
    """

    def __init__(self, kernel_size: int = 3, pad_mode: str = "replicate") -> None:
        """Initialize random convolution augmenter.

        Parameters
        ----------
        kernel_size : int, default=3
            Spatial kernel size. Must be a positive odd integer so padding can
            preserve spatial resolution.
        pad_mode : str, default="replicate"
            Padding mode forwarded to ``torch.nn.functional.pad``.

        Raises
        ------
        ValueError
            If ``kernel_size`` is non-positive or even.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive odd integer, got {self.kernel_size}")
        self.pad_mode = str(pad_mode)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Apply one random convolution kernel to the whole batch.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch with shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Augmented image batch with the same shape as input.
        """
        _check_image_batch(x)
        _, c, _, _ = x.shape
        pad = self.kernel_size // 2

        x_pad = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)
        # One random conv kernel shared across the batch, preserving channel count.
        weight = th.randn((c, c, self.kernel_size, self.kernel_size), device=x.device, dtype=x.dtype)
        weight.mul_(float(c * self.kernel_size * self.kernel_size) ** -0.5)
        return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0)


def svea_augment(obs: th.Tensor, *, aug: RandomConvAug | None = None, kernel_size: int = 3) -> th.Tensor:
    """Apply SVEA random-convolution augmentation to a batch.

    Parameters
    ----------
    obs : torch.Tensor
        Observation batch with shape ``(B, C, H, W)``.
    aug : RandomConvAug, optional
        Reusable augmenter instance.
    kernel_size : int, default=3
        Kernel size used when ``aug`` is not provided.

    Returns
    -------
    torch.Tensor
        Augmented observations with shape identical to ``obs``.
    """
    op = aug if aug is not None else RandomConvAug(kernel_size=kernel_size)
    return op(obs)


def svea_augment_pair(
    obs: th.Tensor,
    next_obs: th.Tensor,
    *,
    aug: RandomConvAug | None = None,
    kernel_size: int = 3,
) -> Tuple[th.Tensor, th.Tensor]:
    """Apply SVEA augmentation to current and next observations.

    Parameters
    ----------
    obs : torch.Tensor
        Current observations with shape ``(B, C, H, W)``.
    next_obs : torch.Tensor
        Next observations with shape ``(B, C, H, W)``.
    aug : RandomConvAug, optional
        Reusable augmenter instance.
    kernel_size : int, default=3
        Kernel size used when ``aug`` is not provided.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple ``(obs_aug, next_obs_aug)``.

    Notes
    -----
    The same augmenter instance is used for both calls when ``aug`` is provided,
    but each forward pass samples independent random kernels.
    """
    op = aug if aug is not None else RandomConvAug(kernel_size=kernel_size)
    return op(obs), op(next_obs)


def svea_mix_loss(loss_clean: th.Tensor, loss_aug: th.Tensor, *, alpha: float = 0.5, beta: float = 0.5) -> th.Tensor:
    """Mix clean and augmented losses in SVEA style.

    Parameters
    ----------
    loss_clean : torch.Tensor
        Loss computed from unaugmented observations.
    loss_aug : torch.Tensor
        Loss computed from augmented observations.
    alpha : float, default=0.5
        Weight for clean loss term.
    beta : float, default=0.5
        Weight for augmented loss term.

    Returns
    -------
    torch.Tensor
        Weighted sum ``alpha * loss_clean + beta * loss_aug``.

    Notes
    -----
    This function does not normalize ``alpha`` and ``beta``; callers can choose
    any weighting scheme as long as it matches their optimization objective.
    """
    return float(alpha) * loss_clean + float(beta) * loss_aug
