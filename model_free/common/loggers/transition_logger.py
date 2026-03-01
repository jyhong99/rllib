"""Transition-level JSONL logging utilities.

This module provides a compact append-only logger for per-transition payloads,
typically used for offline analysis or debugging environment interactions.
"""

from __future__ import annotations

from typing import Any, Mapping

import os

import numpy as np

try:  # pragma: no cover
    import torch as th  # type: ignore
except Exception:  # pragma: no cover
    th = None  # type: ignore

from rllib.model_free.common.utils.logger_utils import _ensure_dir, _json_dumps


def _to_jsonable(x: Any) -> Any:
    """
    Recursively convert common tensor/array containers into JSON-safe objects.

    Parameters
    ----------
    x : Any
        Arbitrary input object that may contain NumPy arrays/scalars, PyTorch
        tensors, nested mappings, or sequences.

    Returns
    -------
    Any
        JSON-serializable equivalent object whenever conversion is supported.
        Unsupported leaf objects are returned unchanged and will be handled by
        the serializer's fallback behavior.

    Notes
    -----
    Conversion rules:
    - ``np.ndarray`` -> Python list via ``tolist()``
    - ``np.floating``/``np.integer`` -> Python scalar via ``item()``
    - ``torch.Tensor`` -> detached CPU list via ``detach().cpu().tolist()``
    - ``Mapping``/``list``/``tuple`` -> recursively converted containers
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if th is not None and isinstance(x, th.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


class TransitionLogger:
    """
    Append-only JSONL logger for env transitions.
    """

    def __init__(self, *, run_dir: str, filename: str = "transitions.jsonl", flush_every: int = 1) -> None:
        """
        Initialize an append-only transition logger.

        Parameters
        ----------
        run_dir : str
            Directory where the transition log file is stored.
        filename : str, default="transitions.jsonl"
            JSONL filename relative to ``run_dir``.
        flush_every : int, default=1
            Flush cadence in number of ``log`` calls. Values below ``1`` are
            clamped to ``1``.

        Notes
        -----
        The output file is opened in append mode to support resume workflows.
        """
        self.run_dir = str(run_dir)
        self.filename = str(filename)
        self.flush_every = max(1, int(flush_every))

        _ensure_dir(self.run_dir)
        self.path = os.path.join(self.run_dir, self.filename)
        self._f = open(self.path, "a", encoding="utf-8")
        self._calls = 0

    def log(self, payload: Mapping[str, Any]) -> None:
        """
        Serialize and append one transition payload line.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Transition dictionary to serialize. Values may include NumPy arrays,
            tensors, nested mappings, and scalar metadata.

        Notes
        -----
        This method is best-effort and intentionally swallows exceptions so
        transition logging cannot interrupt training.
        """
        try:
            s = _json_dumps(_to_jsonable(payload))
            self._f.write(s + "\n")
            self._calls += 1
            if (self._calls % self.flush_every) == 0:
                self._f.flush()
        except Exception:
            return

    def close(self) -> None:
        """
        Flush and close the underlying file handle (best-effort).

        Notes
        -----
        Any exceptions from flush/close are suppressed to keep shutdown paths
        robust.
        """
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass
