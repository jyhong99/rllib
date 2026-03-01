"""Metrics logging callback utilities.

This module provides a callback that forwards update/evaluation metrics to the
trainer logger with optional key filtering, coercion, and rate-limiting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from rllib.model_free.common.callbacks.base_callback import BaseCallback
from rllib.model_free.common.utils.callback_utils import IntervalGate, _coerce_scalar_mapping, _infer_step, _safe_update_step


class MetricsLoggerCallback(BaseCallback):
    """
    Log update/eval metrics dicts via the trainer's logger (best-effort).

    This is a lightweight bridge that forwards metrics provided to callback hooks
    into the trainer's logger with optional filtering, prefixing, and rate limiting.

    Parameters
    ----------
    log_on_update:
        If True, log metrics provided to ``on_update``.
    log_on_eval:
        If True, log metrics provided to ``on_eval_end``.
    log_every_updates:
        Log every N update steps. If ``<= 0``, update logging is disabled.
    log_prefix_update:
        Prefix used for update metrics.
    log_prefix_eval:
        Prefix used for eval metrics.
    keys:
        Optional list of metric keys to keep (applies to update metrics by default).
        If None, all metrics are used.
    eval_keys:
        Optional list of metric keys to keep for eval metrics. If None, falls back to ``keys``.
    drop_non_scalars:
        If True, keep only scalar-like metrics (finite floats) using
        ``_coerce_scalar_mapping``. If False, pass through values as-is.
    """

    def __init__(
        self,
        *,
        log_on_update: bool = True,
        log_on_eval: bool = True,
        log_every_updates: int = 200,
        log_prefix_update: str = "train/",
        log_prefix_eval: str = "eval/",
        keys: Optional[Sequence[str]] = None,
        eval_keys: Optional[Sequence[str]] = None,
        drop_non_scalars: bool = True,
    ) -> None:
        """Initialize metrics forwarding behavior.

        Parameters
        ----------
        log_on_update : bool, default=True
            Whether to log metrics passed to ``on_update``.
        log_on_eval : bool, default=True
            Whether to log metrics passed to ``on_eval_end``.
        log_every_updates : int, default=200
            Update-step interval for update-metric logs.
        log_prefix_update : str, default="train/"
            Prefix used for update metrics.
        log_prefix_eval : str, default="eval/"
            Prefix used for evaluation metrics.
        keys : Sequence[str] | None, default=None
            Optional whitelist for update metric keys.
        eval_keys : Sequence[str] | None, default=None
            Optional whitelist for evaluation metric keys. Falls back to
            ``keys`` when omitted.
        drop_non_scalars : bool, default=True
            Whether to coerce and retain only scalar metrics.
        """
        self.log_on_update = bool(log_on_update)
        self.log_on_eval = bool(log_on_eval)
        self.log_every_updates = int(log_every_updates)
        self.log_prefix_update = str(log_prefix_update)
        self.log_prefix_eval = str(log_prefix_eval)
        self.keys = None if keys is None else [str(k) for k in keys]
        self.eval_keys = None if eval_keys is None else [str(k) for k in eval_keys]
        self.drop_non_scalars = bool(drop_non_scalars)

        self._gate = IntervalGate(every=self.log_every_updates, mode="mod")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _select_metrics(self, metrics: Mapping[str, Any], keys: Optional[Sequence[str]]) -> Dict[str, Any]:
        """
        Select a subset of metrics by key.

        Parameters
        ----------
        metrics:
            Input metrics mapping.
        keys:
            Optional list of keys to select. If None, returns all metrics.

        Returns
        -------
        Dict[str, Any]
            Selected metrics with stringified keys.
        """
        if keys is None:
            return {str(k): v for k, v in metrics.items()}
        return {str(k): metrics[k] for k in keys if k in metrics}

    def _coerce(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Optionally coerce metrics to finite scalars.

        Parameters
        ----------
        metrics:
            Input metrics mapping.

        Returns
        -------
        Dict[str, Any]
            Scalar-only mapping if ``drop_non_scalars=True``, else passthrough.
        """
        if not self.drop_non_scalars:
            return {str(k): v for k, v in metrics.items()}
        return _coerce_scalar_mapping(metrics)

    # ---------------------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------------------
    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Log filtered update metrics when schedule allows.

        Parameters
        ----------
        trainer : Any
            Trainer object used for update-step inference and logging dispatch.
        metrics : Dict[str, Any] | None, default=None
            Update metrics dictionary.

        Returns
        -------
        bool
            Always ``True``. This callback never requests early stop.
        """
        if not self.log_on_update or not metrics or not isinstance(metrics, Mapping):
            return True
        if self.log_every_updates <= 0:
            return True

        upd = _safe_update_step(trainer)
        if upd <= 0:
            return True

        if not self._gate.ready(upd):
            return True

        selected = self._select_metrics(metrics, self.keys)
        payload = self._coerce(selected) if self.drop_non_scalars else selected
        if payload:
            self.log(trainer, payload, step=upd, prefix=self.log_prefix_update)
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """Log filtered evaluation metrics.

        Parameters
        ----------
        trainer : Any
            Trainer object used for step inference and logging dispatch.
        metrics : Dict[str, Any]
            Evaluation metrics dictionary.

        Returns
        -------
        bool
            Always ``True``. This callback never requests early stop.
        """
        if not self.log_on_eval or not metrics or not isinstance(metrics, Mapping):
            return True

        keys = self.eval_keys if self.eval_keys is not None else self.keys
        selected = self._select_metrics(metrics, keys)
        payload = self._coerce(selected) if self.drop_non_scalars else selected
        if payload:
            step = _infer_step(trainer)
            self.log(trainer, payload, step=step, prefix=self.log_prefix_eval)
        return True
