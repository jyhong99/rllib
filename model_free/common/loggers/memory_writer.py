"""In-memory metric writer for tests and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rllib.model_free.common.loggers.base_writer import Writer


@dataclass
class MemoryWriter(Writer):
    """
    In-memory writer for tests or lightweight introspection.

    Stores each row in a list with optional max length (FIFO truncation).
    """

    maxlen: Optional[int] = None
    rows: List[Dict[str, float]] = field(default_factory=list)

    def write(self, row: Dict[str, float]) -> None:
        """
        Append a metric row to memory with optional FIFO truncation.

        Parameters
        ----------
        row : Dict[str, float]
            Scalar metric mapping to store. The input is copied so later caller
            mutations do not affect stored history.

        Notes
        -----
        When ``maxlen`` is set and positive, old rows are dropped from the
        front once capacity is exceeded.
        """
        self.rows.append(dict(row))
        if self.maxlen is not None and self.maxlen > 0:
            excess = len(self.rows) - int(self.maxlen)
            if excess > 0:
                del self.rows[:excess]

    def flush(self) -> None:
        """
        No-op flush implementation for interface compatibility.

        Notes
        -----
        Data is already resident in memory, so there is no buffered I/O to
        flush.
        """
        return

    def close(self) -> None:
        """
        No-op close implementation for interface compatibility.

        Notes
        -----
        This writer intentionally keeps collected rows after close so tests can
        inspect them.
        """
        return
