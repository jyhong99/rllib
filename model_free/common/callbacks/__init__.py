"""
callbacks
====================

Public entrypoint for the callback system.

This package defines:
- The callback API (BaseCallback) and a simple dispatcher (CallbackList)
- A set of standard callbacks used by RL trainers
- A factory utility (build_callbacks) that assembles a default callback stack

Why this file exists
--------------------
Many codebases treat ``callbacks`` as a *stable import surface*:

    from <pkg>.callbacks import build_callbacks, EvalCallback, ...

Over time, implementations may move (e.g., ``build_callbacks`` living in
``callback_builder.py``), but call-sites should not break. Therefore, this
module re-exports the canonical symbols to keep imports stable and prevent
ImportError across refactors.

Optional dependencies / minimal deployments
-------------------------------------------
Some callbacks depend on optional packages (e.g., Ray) or are considered
"non-essential" in minimal deployments. Those are imported in a best-effort
manner:

- If the import succeeds: the symbol is available normally.
- If it fails: the symbol is set to ``None``.

This allows downstream code to do:

    if RayReportCallback is not None:
        ...

instead of requiring conditional imports everywhere.

Export policy
-------------
Only the names listed in ``__all__`` are considered part of the public API.
Internal modules may contain additional helpers that are not exported here.

Notes
-----
- This module performs imports only; it should not contain heavy logic.
- The callback factory is re-exported from ``callback_builder`` to provide a
  single canonical import location.
"""

from __future__ import annotations

from rllib.model_free.common.callbacks.base_callback import BaseCallback, CallbackList

# -----------------------------------------------------------------------------
# Core/standard callbacks (always expected to exist)
# -----------------------------------------------------------------------------
from rllib.model_free.common.callbacks.best_model_callback import BestModelCallback
from rllib.model_free.common.callbacks.checkpoint_callback import CheckpointCallback
from rllib.model_free.common.callbacks.early_stop_callback import EarlyStopCallback
from rllib.model_free.common.callbacks.eval_callback import EvalCallback
from rllib.model_free.common.callbacks.metric_threshold_callback import MetricThresholdCallback
from rllib.model_free.common.callbacks.metrics_logger_callback import MetricsLoggerCallback
from rllib.model_free.common.callbacks.nan_guard_callback import NaNGuardCallback

# -----------------------------------------------------------------------------
# Optional / extra callbacks (best-effort imports)
#
# These may be absent in minimal deployments. When unavailable, the symbol is
# set to None so callers can feature-detect safely.
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from rllib.model_free.common.callbacks.config_and_env_info_callback import ConfigAndEnvInfoCallback
except Exception:  # pragma: no cover
    ConfigAndEnvInfoCallback = None  # type: ignore

try:  # pragma: no cover
    from rllib.model_free.common.callbacks.episode_stats_callback import EpisodeStatsCallback
except Exception:  # pragma: no cover
    EpisodeStatsCallback = None  # type: ignore

try:  # pragma: no cover
    from rllib.model_free.common.callbacks.timing_callback import TimingCallback
except Exception:  # pragma: no cover
    TimingCallback = None  # type: ignore

try:  # pragma: no cover
    from rllib.model_free.common.callbacks.lr_logging_callback import LRLoggingCallback
except Exception:  # pragma: no cover
    LRLoggingCallback = None  # type: ignore

try:  # pragma: no cover
    from rllib.model_free.common.callbacks.grad_param_norm_callback import GradParamNormCallback
except Exception:  # pragma: no cover
    GradParamNormCallback = None  # type: ignore

# -----------------------------------------------------------------------------
# Optional Ray callbacks (best-effort imports)
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from rllib.model_free.common.callbacks.ray_report_callback import RayReportCallback
    from rllib.model_free.common.callbacks.ray_tune_checkpoint_callback import RayTuneCheckpointCallback
except Exception:  # pragma: no cover
    RayReportCallback = None  # type: ignore
    RayTuneCheckpointCallback = None  # type: ignore

# -----------------------------------------------------------------------------
# Factory / builder (canonical import path)
# -----------------------------------------------------------------------------
from rllib.model_free.common.callbacks.callback_builder import build_callbacks


__all__ = [
    # API
    "BaseCallback",
    "CallbackList",
    # Standard callbacks
    "EvalCallback",
    "CheckpointCallback",
    "BestModelCallback",
    "EarlyStopCallback",
    "MetricThresholdCallback",
    "MetricsLoggerCallback",
    "NaNGuardCallback",
    # Optional callbacks
    "ConfigAndEnvInfoCallback",
    "EpisodeStatsCallback",
    "TimingCallback",
    "LRLoggingCallback",
    "GradParamNormCallback",
    "RayReportCallback",
    "RayTuneCheckpointCallback",
    # Factory
    "build_callbacks",
]
