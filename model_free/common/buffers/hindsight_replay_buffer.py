"""Hindsight Experience Replay (HER) buffer utilities.

This module provides a replay buffer that supports goal relabeling during
sampling, allowing sparse-goal tasks to learn from unsuccessful rollouts by
reinterpreting achieved goals as desired goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch as th

from rllib.model_free.common.buffers.base_buffer import BaseReplayBuffer
from rllib.model_free.common.buffers.replay_buffer import ReplayBatch
from rllib.model_free.common.utils.buffer_utils import _uniform_indices
from rllib.model_free.common.utils.common_utils import _to_tensor


@dataclass
class HindsightReplayBatch(ReplayBatch):
    """Batch payload returned by :class:`HindsightReplayBuffer`.

    Extends :class:`ReplayBatch` with HER-specific goal fields and a relabeling
    mask.
    """

    desired_goals: th.Tensor = None  # type: ignore[assignment]
    achieved_goals: th.Tensor = None  # type: ignore[assignment]
    next_achieved_goals: th.Tensor = None  # type: ignore[assignment]
    her_mask: th.Tensor = None  # type: ignore[assignment]


def _ensure_array(
    x: Any,
    *,
    dtype: Any,
    shape: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Convert an input to ndarray and validate dtype/shape.

    Parameters
    ----------
    x : Any
        Input array-like object.
    dtype : Any
        Target numpy dtype.
    shape : tuple[int, ...]
        Required exact shape.
    name : str
        Field name used in validation errors.

    Returns
    -------
    np.ndarray
        Converted array with target dtype and validated shape.

    Raises
    ------
    ValueError
        If the converted array shape differs from ``shape``.
    """
    target_dtype = np.dtype(dtype)
    if isinstance(x, np.ndarray):
        if x.dtype != target_dtype:
            x = x.astype(target_dtype, copy=False)
    else:
        x = np.asarray(x, dtype=target_dtype)
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {x.shape}")
    return x


class HindsightReplayBuffer(BaseReplayBuffer):
    """Replay buffer with HER relabeling.

    Parameters
    ----------
    capacity : int
        Max number of transitions.
    obs_shape : tuple[int, ...]
        Observation shape.
    action_shape : tuple[int, ...]
        Action shape.
    goal_shape : tuple[int, ...]
        Goal vector shape.
    reward_fn : callable
        Reward recomputation callable:
        ``reward_fn(next_achieved_goal, desired_goal) -> float or ndarray``.
    done_fn : callable or None, default=None
        Optional termination recomputation callable:
        ``done_fn(next_achieved_goal, desired_goal) -> bool/float``.
    her_ratio : float, default=0.8
        Probability of relabeling each sampled transition.
    device : str or torch.device, default="cpu"
        Output tensor device.
    dtype_obs : Any, default=np.float32
        Observation dtype.
    dtype_act : Any, default=np.float32
        Action dtype.
    dtype_goal : Any, default=np.float32
        Goal dtype.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        *,
        goal_shape: Tuple[int, ...],
        reward_fn: Callable[[np.ndarray, np.ndarray], Any],
        done_fn: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        her_ratio: float = 0.8,
        device: Union[str, th.device] = "cpu",
        dtype_obs: Any = np.float32,
        dtype_act: Any = np.float32,
        dtype_goal: Any = np.float32,
    ) -> None:
        """Initialize HER replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of transitions stored.
        obs_shape : tuple[int, ...]
            Observation shape excluding batch dimension.
        action_shape : tuple[int, ...]
            Action shape excluding batch dimension.
        goal_shape : tuple[int, ...]
            Goal tensor shape.
        reward_fn : Callable[[np.ndarray, np.ndarray], Any]
            Reward recomputation callback used for relabeled goals.
        done_fn : Callable[[np.ndarray, np.ndarray], Any] | None, default=None
            Optional done recomputation callback for relabeled goals.
        her_ratio : float, default=0.8
            Probability of relabeling each sampled transition.
        device : str | torch.device, default="cpu"
            Device used when returning sampled tensors.
        dtype_obs : Any, default=np.float32
            Observation storage dtype.
        dtype_act : Any, default=np.float32
            Action storage dtype.
        dtype_goal : Any, default=np.float32
            Goal storage dtype.

        Raises
        ------
        TypeError
            If ``reward_fn`` is not callable or ``done_fn`` is provided but not
            callable.
        ValueError
            If ``her_ratio`` is outside ``[0, 1]``.
        """
        if not callable(reward_fn):
            raise TypeError("reward_fn must be callable")
        if done_fn is not None and not callable(done_fn):
            raise TypeError("done_fn must be callable when provided")
        if not (0.0 <= float(her_ratio) <= 1.0):
            raise ValueError(f"her_ratio must be in [0,1], got {her_ratio}")

        self.goal_shape = tuple(goal_shape)
        self.dtype_goal = dtype_goal
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.her_ratio = float(her_ratio)

        super().__init__(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            dtype_obs=dtype_obs,
            dtype_act=dtype_act,
        )

        self.desired_goals = np.zeros((self.capacity, *self.goal_shape), dtype=self.dtype_goal)
        self.achieved_goals = np.zeros((self.capacity, *self.goal_shape), dtype=self.dtype_goal)
        self.next_achieved_goals = np.zeros((self.capacity, *self.goal_shape), dtype=self.dtype_goal)

        # Monotonic insertion timestamp for temporal-contiguity checks.
        self._time_index = np.full((self.capacity,), -1, dtype=np.int64)
        self._insert_count = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
        next_achieved_goal: np.ndarray,
        **_: Any,
    ) -> None:
        """Insert one transition with goal fields."""
        obs = _ensure_array(obs, dtype=self.dtype_obs, shape=self.obs_shape, name="obs")
        next_obs = _ensure_array(next_obs, dtype=self.dtype_obs, shape=self.obs_shape, name="next_obs")
        action = _ensure_array(action, dtype=self.dtype_act, shape=self.action_shape, name="action")
        desired_goal = _ensure_array(
            desired_goal, dtype=self.dtype_goal, shape=self.goal_shape, name="desired_goal"
        )
        achieved_goal = _ensure_array(
            achieved_goal, dtype=self.dtype_goal, shape=self.goal_shape, name="achieved_goal"
        )
        next_achieved_goal = _ensure_array(
            next_achieved_goal, dtype=self.dtype_goal, shape=self.goal_shape, name="next_achieved_goal"
        )

        idx = self.pos
        self._time_index[idx] = int(self._insert_count)
        self._insert_count += 1

        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx, 0] = float(reward)
        self.next_observations[idx] = next_obs
        self.dones[idx, 0] = 1.0 if bool(done) else 0.0

        self.desired_goals[idx] = desired_goal
        self.achieved_goals[idx] = achieved_goal
        self.next_achieved_goals[idx] = next_achieved_goal

        self._advance()

    def _future_candidates(self, idx: int) -> np.ndarray:
        """Return valid future indices in the same contiguous episode segment."""
        cap = int(self.capacity)
        out: list[int] = []
        cur = int(idx)
        for _ in range(cap - 1):
            if float(self.dones[cur, 0]) > 0.5:
                break
            nxt = (cur + 1) % cap
            if int(self._time_index[cur]) < 0 or int(self._time_index[nxt]) != int(self._time_index[cur]) + 1:
                break
            out.append(int(nxt))
            cur = nxt
        return np.asarray(out, dtype=np.int64)

    def _sample_her_goals(self, idx: np.ndarray, her_mask: np.ndarray) -> np.ndarray:
        """Relabel desired goals by sampling a future achieved goal."""
        desired = self.desired_goals[idx].copy()
        if not np.any(her_mask):
            return desired

        for j, i in enumerate(idx.tolist()):
            if not bool(her_mask[j]):
                continue
            fut = self._future_candidates(int(i))
            if fut.size <= 0:
                continue
            k = int(np.random.randint(0, fut.size))
            desired[j] = self.achieved_goals[int(fut[k])]
        return desired

    def sample(
        self,
        batch_size: int,
        *,
        her_ratio: Optional[float] = None,
        **_: Any,
    ) -> HindsightReplayBatch:
        """Sample batch and apply HER relabeling."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty HindsightReplayBuffer.")

        ratio = self.her_ratio if her_ratio is None else float(her_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"her_ratio must be in [0,1], got {ratio}")

        idx = _uniform_indices(self.size, batch_size)
        her_mask_np = (np.random.rand(int(batch_size)) < ratio).astype(np.float32).reshape(-1, 1)

        desired = self._sample_her_goals(idx, her_mask_np.reshape(-1).astype(bool))
        achieved = self.achieved_goals[idx]
        next_achieved = self.next_achieved_goals[idx]

        rew = self.rewards[idx].copy()
        don = self.dones[idx].copy()
        for j in range(int(batch_size)):
            if her_mask_np[j, 0] <= 0.0:
                continue
            rj = self.reward_fn(next_achieved[j], desired[j])
            rew[j, 0] = float(np.asarray(rj, dtype=np.float32).reshape(()))
            if self.done_fn is not None:
                dj = self.done_fn(next_achieved[j], desired[j])
                don[j, 0] = 1.0 if bool(np.asarray(dj).reshape(())) else 0.0

        return HindsightReplayBatch(
            observations=_to_tensor(self.observations[idx], device=self.device),
            actions=_to_tensor(self.actions[idx], device=self.device),
            rewards=_to_tensor(rew, device=self.device),
            next_observations=_to_tensor(self.next_observations[idx], device=self.device),
            dones=_to_tensor(don, device=self.device),
            desired_goals=_to_tensor(desired, device=self.device),
            achieved_goals=_to_tensor(achieved, device=self.device),
            next_achieved_goals=_to_tensor(next_achieved, device=self.device),
            her_mask=_to_tensor(her_mask_np, device=self.device),
        )
