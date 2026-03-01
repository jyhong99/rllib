"""Console metric writer for lightweight human-readable output."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from rllib.model_free.common.loggers.base_writer import Writer
from rllib.model_free.common.utils.logger_utils import _get_step, _split_meta


class StdoutWriter(Writer):
    """
    Simple stdout writer for metric rows.

    This is a lightweight writer backend that prints a compact, human-readable
    line for each write call. It can be useful in minimal setups where you want
    console-only logging without CSV/JSON/TensorBoard.

    Parameters
    ----------
    every:
        Print every N writes based on step if available; if <= 0, disables output.
    keys:
        Optional sequence of metric keys to display (after meta split). If None,
        shows a small subset of available metrics.
    max_items:
        Maximum number of metric items to print when `keys` is None.
    """

    def __init__(
        self,
        *,
        every: int = 1,
        keys: Optional[Sequence[str]] = None,
        max_items: int = 6,
    ) -> None:
        """
        Initialize a stdout writer.

        Parameters
        ----------
        every : int, default=1
            Output cadence. When a valid step is present, logging occurs when
            ``step % every == 0``. Without a valid step, cadence falls back to
            call count. Values ``<= 0`` disable output.
        keys : Sequence[str], optional
            Explicit metric keys to print (in order). Missing keys are skipped.
            When omitted, the writer prints up to ``max_items`` keys from the
            metric row.
        max_items : int, default=6
            Maximum number of metrics shown when ``keys`` is not provided.
        """
        self.every = int(every)
        self.keys = None if keys is None else [str(k) for k in keys]
        self.max_items = int(max_items)
        self._calls = 0

    def write(self, row: Dict[str, float]) -> None:
        """
        Print one compact metric line to stdout if cadence conditions pass.

        Parameters
        ----------
        row : Dict[str, float]
            Input row containing meta keys (for example ``step``) and scalar
            metric values.

        Notes
        -----
        The rendered format is:
        ``[step=<int>] key1=value1 key2=value2 ...``
        """
        if self.every <= 0:
            return

        step = _get_step(row)
        self._calls += 1

        # Prefer step-based cadence when a valid step is present.
        if step > 0 and (step % self.every) != 0:
            return
        if step <= 0 and (self._calls % self.every) != 0:
            return

        meta, metrics = _split_meta(row)

        # Select keys
        items: Iterable[str]
        if self.keys is not None:
            items = [k for k in self.keys if k in metrics]
        else:
            items = list(metrics.keys())[: max(0, self.max_items)]

        parts = []
        for k in items:
            try:
                parts.append(f"{k}={float(metrics[k]):.4g}")
            except Exception:
                continue

        step_val = int(meta.get("step", step))
        msg = f"[step={step_val}] " + " ".join(parts)
        print(msg)

    def flush(self) -> None:
        """
        No-op flush for compatibility with the writer interface.
        """
        return

    def close(self) -> None:
        """
        No-op close for compatibility with the writer interface.
        """
        return
