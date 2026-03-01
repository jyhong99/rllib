"""DrQ image-space augmentations.

This module implements random-shift augmentation used by DrQ/DrQ-v2 style
pixel-based reinforcement learning agents. The primary objective is to improve
data efficiency and reduce overfitting to exact pixel alignments by sampling
small spatial translations per image in a batch.

The implementation is designed to be:

- differentiable (via ``torch.nn.functional.grid_sample``),
- shape-preserving (input/output are both ``(B, C, H, W)``),
- reusable through lightweight helper functions for single and paired batches.
"""

from __future__ import annotations

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """Random-shift image augmentation for batched observations.

    Parameters
    ----------
    pad : int, default=4
        Number of pixels to replicate-pad on each image border prior to
        sampling a shifted crop. Effective shift range is
        ``[0, 2 * pad]`` pixels along each spatial axis.

    Notes
    -----
    The transform uses normalized sampling coordinates and bilinear resampling.
    For each batch item, an independent random translation is sampled while a
    deterministic base grid is reused.
    """

    def __init__(self, pad: int = 4) -> None:
        super().__init__()
        self.pad = int(pad)
        if self.pad < 0:
            raise ValueError(f"pad must be >= 0, got {self.pad}")
        self._grid_cache: dict[tuple[int, int, int, str, str], th.Tensor] = {}

    def _base_grid(self, *, b: int, h: int, w: int, device: th.device, dtype: th.dtype) -> th.Tensor:
        """Build or reuse a base sampling grid for a given batch shape/device.

        Parameters
        ----------
        b : int
            Batch size.
        h : int
            Input height.
        w : int
            Input width.
        device : torch.device
            Target device of returned grid.
        dtype : torch.dtype
            Target dtype of returned grid.

        Returns
        -------
        torch.Tensor
            Base grid with shape ``(B, H, W, 2)``.
        """
        key = (b, h, w, str(device), str(dtype))
        cached = self._grid_cache.get(key, None)
        if cached is not None:
            return cached

        eps_h = 2.0 / float(h + 2 * self.pad)
        eps_w = 2.0 / float(w + 2 * self.pad)
        yy, xx = th.meshgrid(
            th.linspace(-1.0 + eps_h, 1.0 - eps_h, h, device=device, dtype=dtype),
            th.linspace(-1.0 + eps_w, 1.0 - eps_w, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base = th.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        self._grid_cache[key] = base
        return base

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Apply random-shift augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch with shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Augmented image batch with the same shape as ``x``.

        Raises
        ------
        ValueError
            If ``x`` does not have rank 4.
        """
        if x.ndim != 4:
            raise ValueError(f"expected (B,C,H,W), got shape={tuple(x.shape)}")
        if self.pad == 0:
            return x

        b, _, h, w = x.shape
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        base_grid = self._base_grid(b=b, h=h, w=w, device=x.device, dtype=x.dtype)

        shift_x = th.randint(0, 2 * self.pad + 1, (b, 1, 1, 1), device=x.device).to(x.dtype)
        shift_y = th.randint(0, 2 * self.pad + 1, (b, 1, 1, 1), device=x.device).to(x.dtype)
        shift = th.cat(
            (
                shift_x * (2.0 / float(w + 2 * self.pad)),
                shift_y * (2.0 / float(h + 2 * self.pad)),
            ),
            dim=-1,
        )
        grid = base_grid + shift
        return F.grid_sample(x_pad, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def drq_augment(obs: th.Tensor, *, pad: int = 4, aug: RandomShiftsAug | None = None) -> th.Tensor:
    """Apply DrQ random-shift augmentation to one batch.

    Parameters
    ----------
    obs : torch.Tensor
        Observation tensor with shape ``(B, C, H, W)``.
    pad : int, default=4
        Replicate-padding size used when ``aug`` is not provided.
    aug : RandomShiftsAug, optional
        Reusable augmenter instance. Passing an instance avoids repeated module
        construction and enables grid caching across calls.

    Returns
    -------
    torch.Tensor
        Augmented observations with shape identical to ``obs``.
    """
    op = aug if aug is not None else RandomShiftsAug(pad=pad)
    return op(obs)


def drq_augment_pair(
    obs: th.Tensor,
    next_obs: th.Tensor,
    *,
    pad: int = 4,
    aug: RandomShiftsAug | None = None,
) -> Tuple[th.Tensor, th.Tensor]:
    """Apply DrQ augmentation independently to ``obs`` and ``next_obs``.

    Parameters
    ----------
    obs : torch.Tensor
        Current observations ``(B, C, H, W)``.
    next_obs : torch.Tensor
        Next observations ``(B, C, H, W)``.
    pad : int, default=4
        Replicate-padding size used when ``aug`` is not provided.
    aug : RandomShiftsAug, optional
        Reusable augmenter instance.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple ``(obs_aug, next_obs_aug)``.
    """
    op = aug if aug is not None else RandomShiftsAug(pad=pad)
    return op(obs), op(next_obs)
