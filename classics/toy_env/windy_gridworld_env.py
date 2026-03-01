from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .tabular_grid_base import _TabularGridBase


class WindyGridworldEnv(_TabularGridBase):
    """
    Sutton-style windy gridworld (7x10 by default).

    Parameters
    ----------
    height : int, default=7
        Number of rows.
    width : int, default=10
        Number of columns.
    start : Tuple[int, int], default=(3, 0)
        Start position.
    goal : Tuple[int, int], default=(3, 7)
        Goal position.
    wind_strength : Optional[Sequence[int]], default=None
        Upward push per column. If ``None``, uses Sutton's canonical profile.
    max_steps : Optional[int], default=300
        Episode time limit.

    Notes
    -----
    Wind is applied after the action move and uses the destination column.
    """

    def __init__(
        self,
        height: int = 7,
        width: int = 10,
        start: Tuple[int, int] = (3, 0),
        goal: Tuple[int, int] = (3, 7),
        wind_strength: Optional[Sequence[int]] = None,
        max_steps: Optional[int] = 300,
    ) -> None:
        super().__init__(height=height, width=width, max_steps=max_steps)
        if wind_strength is None:
            wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        if len(wind_strength) != self.width:
            raise ValueError("wind_strength length must match width")
        if any(int(w) < 0 for w in wind_strength):
            raise ValueError("wind_strength values must be >= 0")

        self.start = self._clip(*start)
        self.goal = self._clip(*goal)
        self.wind = tuple(int(w) for w in wind_strength)

    def _start_pos(self) -> Tuple[int, int]:
        return self.start

    def _reset_info(self) -> Dict[str, Any]:
        return {"start": self.start, "goal": self.goal, "wind": self.wind}

    def _transition(self, action: int):
        row, col = self._move(self._row, self._col, action)

        # Wind pushes upward (toward row 0) according to destination column.
        row = max(0, row - self.wind[col])

        terminated = (row, col) == self.goal
        reward = -1.0
        info = {
            "row": row,
            "col": col,
            "goal_reached": terminated,
            "wind_applied": self.wind[col],
        }
        return row, col, reward, terminated, info
