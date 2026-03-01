"""Metric-threshold stop callback.

This module provides a callback that requests training stop once an evaluation
metric crosses a configured threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from rllib.model_free.common.callbacks.base_callback import BaseCallback
from rllib.model_free.common.utils.callback_utils import _safe_env_step, _to_finite_float


class MetricThresholdCallback(BaseCallback):
    """
    Stop training when an evaluation metric crosses a threshold.

    This is commonly used to stop training once a target return or success rate
    is achieved (e.g., "stop when eval_return_mean >= 200").

    Parameters
    ----------
    metric_key:
        Key to read from evaluation metrics dict.
    threshold:
        Target threshold. If None or non-finite, the callback is disabled.
    mode:
        "max" -> stop when value >= threshold
        "min" -> stop when value <= threshold
    log_prefix:
        Prefix used for logging when the threshold is triggered.
    """

    def __init__(
        self,
        metric_key: str = "eval_return_mean",
        threshold: Optional[float] = None,
        mode: Literal["max", "min"] = "max",
        *,
        log_prefix: str = "sys/",
    ) -> None:
        """Initialize threshold-based stopping rule.

        Parameters
        ----------
        metric_key : str, default="eval_return_mean"
            Evaluation metric key to monitor.
        threshold : float | None, default=None
            Target threshold. ``None`` disables trigger checks.
        mode : {"max", "min"}, default="max"
            Threshold direction. ``"max"`` triggers on ``value >= threshold``;
            ``"min"`` triggers on ``value <= threshold``.
        log_prefix : str, default="sys/"
            Prefix for threshold-trigger logs.

        Raises
        ------
        ValueError
            If ``mode`` is not ``"max"`` or ``"min"``.
        """
        self.metric_key = str(metric_key)
        self.threshold = threshold
        self.mode: Literal["max", "min"] = mode
        self.log_prefix = str(log_prefix)

        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode}")

    def _is_triggered(self, value: float, threshold: float) -> bool:
        """
        Check whether the threshold condition is met.

        Parameters
        ----------
        value:
            Current metric value.
        threshold:
            Threshold to compare against.

        Returns
        -------
        bool
            True if the stop condition is triggered.
        """
        if self.mode == "max":
            return value >= threshold
        return value <= threshold

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """Check metric threshold condition at evaluation end.

        Parameters
        ----------
        trainer : Any
            Trainer object used only for logging and step inference.
        metrics : Dict[str, Any]
            Evaluation metrics dictionary containing ``metric_key``.

        Returns
        -------
        bool
            ``False`` when threshold condition is triggered, otherwise ``True``.
        """
        if not isinstance(metrics, dict):
            return True

        if self.threshold is None:
            return True

        thr = _to_finite_float(self.threshold)
        if thr is None:
            return True

        val = _to_finite_float(metrics.get(self.metric_key, None))
        if val is None:
            return True

        if self._is_triggered(val, thr):
            self.log(
                trainer,
                {
                    "threshold/triggered": 1.0,
                    "threshold/value": float(val),
                    "threshold/threshold": float(thr),
                    "threshold/mode_max": 1.0 if self.mode == "max" else 0.0,
                },
                step=_safe_env_step(trainer),
                prefix=self.log_prefix,
            )
            return False

        return True
