from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .tabular_grid_base import _TabularGridBase


class GridworldEnv(_TabularGridBase):
    """
    Deterministic tabular gridworld with one absorbing goal state.

    Parameters
    ----------
    height : int, default=5
        Number of rows.
    width : int, default=5
        Number of columns.
    start : Tuple[int, int], default=(0, 0)
        Start position.
    goal : Optional[Tuple[int, int]], default=None
        Goal position. If ``None``, uses bottom-right corner.
    step_reward : float, default=-1.0
        Reward for non-terminal transitions.
    goal_reward : float, default=0.0
        Reward upon entering goal state.
    max_steps : Optional[int], default=None
        Episode time limit.
    """

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        start: Tuple[int, int] = (0, 0),
        goal: Optional[Tuple[int, int]] = None,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__(height=height, width=width, max_steps=max_steps)
        if goal is None:
            goal = (height - 1, width - 1)

        self.start = self._clip(*start)
        self.goal = self._clip(*goal)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)

    def _start_pos(self) -> Tuple[int, int]:
        return self.start

    def _reset_info(self) -> Dict[str, Any]:
        return {"start": self.start, "goal": self.goal}

    def _transition(self, action: int):
        next_row, next_col = self._move(self._row, self._col, action)
        terminated = (next_row, next_col) == self.goal
        reward = self.goal_reward if terminated else self.step_reward
        info = {
            "row": next_row,
            "col": next_col,
            "goal_reached": terminated,
        }
        return next_row, next_col, reward, terminated, info
