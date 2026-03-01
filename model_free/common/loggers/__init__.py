"""Public logger package API.

This package exposes a composable logging stack:

- :class:`Logger` frontend for metric normalization, buffering, and cadence
- writer backends for CSV, JSONL, TensorBoard, W&B, stdout, and memory
- :func:`build_logger` factory for common backend wiring

Notes
-----
Typical usage builds a logger via :func:`build_logger`, emits scalar mappings
through :meth:`Logger.log`, and closes the logger at process shutdown.

Examples
--------
>>> from rllib.model_free.common.loggers import build_logger
>>> logger = build_logger(log_dir="./runs", exp_name="exp1", use_csv=True)
>>> logger.log({"train/loss": 0.1}, step=1)
>>> logger.close()
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Core logger
# -----------------------------------------------------------------------------
from rllib.model_free.common.loggers.logger import Logger

# -----------------------------------------------------------------------------
# Writer base + concrete writers
# -----------------------------------------------------------------------------
from rllib.model_free.common.loggers.base_writer import Writer, SafeWriter
from rllib.model_free.common.loggers.csv_writer import CSVWriter
from rllib.model_free.common.loggers.jsonl_writer import JSONLWriter
from rllib.model_free.common.loggers.tensorboard_writer import TensorBoardWriter
from rllib.model_free.common.loggers.wandb_writer import WandBWriter
from rllib.model_free.common.loggers.stdout_writer import StdoutWriter
from rllib.model_free.common.loggers.memory_writer import MemoryWriter

# -----------------------------------------------------------------------------
# Builder utility
# -----------------------------------------------------------------------------
from rllib.model_free.common.loggers.logger_builder import build_logger

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # core
    "Logger",

    # writer base
    "Writer",
    "SafeWriter",

    # writers
    "CSVWriter",
    "JSONLWriter",
    "TensorBoardWriter",
    "WandBWriter",
    "StdoutWriter",
    "MemoryWriter",

    # builder
    "build_logger",
]
