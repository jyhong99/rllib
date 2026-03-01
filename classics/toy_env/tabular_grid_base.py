from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        spaces = None  # type: ignore


class _TabularGridBase((gym.Env if gym is not None else object)):  # type: ignore[misc]
    """
    Shared tabular-grid base for deterministic finite MDP environments.

    Parameters
    ----------
    height : int
        Number of grid rows.
    width : int
        Number of grid columns.
    max_steps : Optional[int], default=None
        Episode time limit. If ``None``, defaults to ``height * width * 4``.
    """

    metadata = {"render_modes": []}

    # Action mapping: 0=up, 1=right, 2=down, 3=left
    _ACTION_DELTA = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(self, height: int, width: int, max_steps: Optional[int] = None) -> None:
        """
        Construct the base grid state/action spaces and horizon.

        Raises
        ------
        ValueError
            If grid dimensions or max_steps are invalid.
        ImportError
            If neither Gymnasium nor Gym is installed.
        """
        if spaces is None:  # pragma: no cover
            raise ImportError("gymnasium or gym is required to use tabular grid envs")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be > 0")

        self.height = int(height)
        self.width = int(width)
        self.n_states = self.height * self.width

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)

        default_horizon = self.height * self.width * 4
        self.max_steps = int(max_steps) if max_steps is not None else default_horizon
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self._step_count = 0
        self._row = 0
        self._col = 0

    def _to_state(self, row: int, col: int) -> int:
        """Convert ``(row, col)`` coordinates to tabular state index."""
        return row * self.width + col

    def _clip(self, row: int, col: int) -> Tuple[int, int]:
        """Clamp coordinates to valid grid bounds."""
        return max(0, min(row, self.height - 1)), max(0, min(col, self.width - 1))

    def _move(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """Apply one cardinal action and clamp to grid bounds."""
        d_row, d_col = self._ACTION_DELTA[action]
        return self._clip(row + d_row, col + d_col)

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Gym compatibility seed hook.

        Parameters
        ----------
        seed : Optional[int], default=None
            Unused for deterministic dynamics.
        """
        _ = seed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset to subclass-defined start state.

        Parameters
        ----------
        seed : Optional[int], default=None
            Accepted for API compatibility.
        options : Optional[Dict[str, Any]], default=None
            Accepted for API compatibility; currently unused.

        Returns
        -------
        obs : int
            Discrete state index for the start cell.
        info : Dict[str, Any]
            Environment-specific reset metadata.
        """
        _ = options
        self.seed(seed)
        self._step_count = 0
        self._row, self._col = self._start_pos()
        return int(self._to_state(self._row, self._col)), self._reset_info()

    def step(self, action: int):
        """
        Apply one environment transition.

        Parameters
        ----------
        action : int
            Action id in ``{0, 1, 2, 3}`` corresponding to up/right/down/left.

        Returns
        -------
        obs : int
            Discrete state index after transition.
        reward : float
            Transition reward.
        terminated : bool
            Whether terminal condition was reached.
        truncated : bool
            Whether episode ended due to max step limit.
        info : Dict[str, Any]
            Transition diagnostics, with ``step_count`` added by base class.

        Raises
        ------
        ValueError
            If action is outside valid range.
        """
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"action must be in [0, {self.action_space.n - 1}], got {action}")

        self._step_count += 1
        self._row, self._col, reward, terminated, info = self._transition(action)
        truncated = bool((not terminated) and (self._step_count >= self.max_steps))
        info["step_count"] = self._step_count
        return int(self._to_state(self._row, self._col)), float(reward), bool(terminated), truncated, info

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize grid position and episode counters.

        Returns
        -------
        Dict[str, Any]
            Serializable state payload.
        """
        return {
            "row": int(self._row),
            "col": int(self._col),
            "step_count": int(self._step_count),
            "height": self.height,
            "width": self.width,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore serialized grid position and counters.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary produced by :meth:`state_dict`.
        """
        self._row = int(state.get("row", 0))
        self._col = int(state.get("col", 0))
        self._step_count = int(state.get("step_count", 0))

    def _start_pos(self) -> Tuple[int, int]:
        raise NotImplementedError

    def _reset_info(self) -> Dict[str, Any]:
        return {}

    def _transition(self, action: int) -> Tuple[int, int, float, bool, Dict[str, Any]]:
        raise NotImplementedError

