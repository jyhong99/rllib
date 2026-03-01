from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .tabular_grid_base import _TabularGridBase


class CliffWalkingEnv(_TabularGridBase):
    """
    Sutton-style cliff walking (4x12 by default).

    Parameters
    ----------
    height : int, default=4
        Number of rows.
    width : int, default=12
        Number of columns.
    max_steps : Optional[int], default=200
        Episode time limit.

    Notes
    -----
    Start is bottom-left and goal is bottom-right. The cliff spans the bottom
    row between start and goal. Entering cliff gives reward ``-100`` and resets
    to start without terminating.
    """

    def __init__(self, height: int = 4, width: int = 12, max_steps: Optional[int] = 200) -> None:
        if height < 2 or width < 3:
            raise ValueError("CliffWalking requires at least height>=2 and width>=3")
        super().__init__(height=height, width=width, max_steps=max_steps)
        self.start = (self.height - 1, 0)
        self.goal = (self.height - 1, self.width - 1)

        self._cliff = {(self.height - 1, c) for c in range(1, self.width - 1)}

    def _start_pos(self) -> Tuple[int, int]:
        return self.start

    def _reset_info(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "goal": self.goal,
            "n_cliff_cells": len(self._cliff),
        }

    def _transition(self, action: int):
        next_row, next_col = self._move(self._row, self._col, action)

        fell_off_cliff = (next_row, next_col) in self._cliff
        if fell_off_cliff:
            next_row, next_col = self.start
            reward = -100.0
            terminated = False
        else:
            terminated = (next_row, next_col) == self.goal
            reward = -1.0

        info = {
            "row": next_row,
            "col": next_col,
            "fell_off_cliff": fell_off_cliff,
            "goal_reached": terminated,
        }
        return next_row, next_col, reward, terminated, info
