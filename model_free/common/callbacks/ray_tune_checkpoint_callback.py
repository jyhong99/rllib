"""Ray checkpoint reporting callback.

This module provides best-effort forwarding of trainer-created checkpoints to
Ray AIR / Ray Tune so schedulers and trial analysis can track checkpoints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rllib.model_free.common.callbacks.base_callback import BaseCallback
from rllib.model_free.common.utils.callback_utils import _safe_env_step, _safe_update_step


class RayTuneCheckpointCallback(BaseCallback):
    """
    Bridge Trainer-created checkpoints to Ray Tune / Ray AIR (best-effort).

    This callback is meant to be used when your Trainer is responsible for creating
    checkpoints (typically directories), but you are running the training loop under
    Ray Tune / Ray AIR and want Ray to "see" and track those checkpoints.

    The callback listens to ``on_checkpoint(trainer, path)`` events and attempts to
    report the checkpoint to Ray using whichever reporting backend is available:

    1) Ray AIR (preferred): ``ray.air.session.report(metrics, checkpoint=Checkpoint)``
    2) Legacy Ray Tune:     ``ray.tune.report(..., checkpoint=Checkpoint)``
    3) Otherwise: no-op

    Design principles
    -----------------
    - No persistent Ray context/handles stored on the callback instance.
    - Safe to import and use even when Ray is not installed (all Ray imports are
      local and wrapped in try/except).
    - Best-effort behavior: never raises; never stops training.

    Parameters
    ----------
    report_empty_metrics : bool, default=True
        If True, the callback will still report the checkpoint even when no
        additional scalar metrics are available. This is useful for enabling
        checkpoint-based selection/retention in Ray without requiring a metric.

        Notes:
        - Ray AIR accepts an empty metrics dict.
        - For legacy Tune, we call ``tune.report(**metrics, checkpoint=...)``; an
          empty dict results in reporting only the checkpoint.
    include_steps : bool, default=True
        If True, attach global step counters (if present on the trainer) to the
        metrics payload to avoid reporting an empty dict.

    Notes
    -----
    Trainer contract (duck-typed)
    -----------------------------
    The Trainer is expected to call::

        callbacks.on_checkpoint(trainer, path)

    where:
      - ``path`` is a directory path containing the checkpoint artifacts.

    This callback does not attempt to create or manage checkpoints itself.
    """

    def __init__(
        self,
        *,
        report_empty_metrics: bool = True,
        include_steps: bool = True,
        report_every_saves: int = 1,
        metric_key: Optional[str] = "eval/return_mean",
    ) -> None:
        """Initialize Ray checkpoint-reporting behavior.

        Parameters
        ----------
        report_empty_metrics : bool, default=True
            Whether to report checkpoints even when no extra scalar metrics are
            available.
        include_steps : bool, default=True
            Whether to include global env/update step counters in report payloads.
        report_every_saves : int, default=1
            Emit Ray checkpoint report every N trainer checkpoint saves.
            Non-positive values disable reporting.
        metric_key : str | None, default="eval/return_mean"
            Optional metric key to include from latest evaluation payload when
            available on trainer as ``_last_eval_metrics`` or
            ``last_eval_metrics``.
        """
        self.report_every_saves = int(report_every_saves)
        self.metric_key = None if metric_key is None else str(metric_key)
        self.report_empty_metrics = bool(report_empty_metrics)
        self.include_steps = bool(include_steps)
        self._num_saves = 0

    @staticmethod
    def _maybe_add_steps(trainer: Any, metrics: Dict[str, Any]) -> None:
        """
        Best-effort add global step counters to metrics.

        Parameters
        ----------
        trainer : Any
            Trainer-like object (duck-typed).
        metrics : Dict[str, Any]
            Mutable metrics dict to be updated in-place.
        """
        try:
            metrics.setdefault("sys/global_env_step", float(_safe_env_step(trainer)))
        except Exception:
            pass
        try:
            metrics.setdefault("sys/global_update_step", float(_safe_update_step(trainer)))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers (all best-effort, no persistent state)
    # ------------------------------------------------------------------
    @staticmethod
    def _checkpoint_from_directory(path: str) -> Optional[Any]:
        """
        Create a Ray AIR Checkpoint from a directory (best-effort).

        Parameters
        ----------
        path : str
            Directory path of the saved checkpoint.

        Returns
        -------
        Optional[Any]
            - A ``ray.air.Checkpoint`` object if Ray is available and the path is valid.
            - None if Ray is unavailable, import fails, or path is invalid.

        Notes
        -----
        - Uses ``Checkpoint.from_directory(path)`` which expects a directory layout
          created by the Trainer.
        - This helper intentionally does not validate filesystem existence to keep it
          lightweight and avoid extra IO; failures are handled by Ray import/call errors.
        """
        if not isinstance(path, str) or not path:
            return None
        try:
            from ray.air import Checkpoint  # type: ignore

            return Checkpoint.from_directory(path)
        except Exception:
            return None

    @staticmethod
    def _report_air(metrics: Dict[str, Any], checkpoint: Any) -> bool:
        """
        Report (metrics, checkpoint) to Ray AIR (best-effort).

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics to report along with the checkpoint (often empty).
        checkpoint : Any
            A Ray AIR ``Checkpoint`` object.

        Returns
        -------
        bool
            True if reporting appears successful, False otherwise.

        Notes
        -----
        - Uses ``ray.air.session.report(metrics, checkpoint=checkpoint)``.
        - Swallows all exceptions (Ray not installed, session missing, runtime errors).
        """
        try:
            from ray.air import session  # type: ignore

            session.report(metrics, checkpoint=checkpoint)
            return True
        except Exception:
            return False

    @staticmethod
    def _report_legacy(metrics: Dict[str, Any], checkpoint: Any) -> bool:
        """
        Report (metrics, checkpoint) to legacy Ray Tune (best-effort).

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics to report. Keys must be strings for legacy Tune.
        checkpoint : Any
            A Ray AIR ``Checkpoint`` object.

        Returns
        -------
        bool
            True if reporting appears successful, False otherwise.

        Notes
        -----
        - Uses ``ray.tune.report(**metrics, checkpoint=checkpoint)``.
        - Some Tune versions historically accepted ``checkpoint=...``; if the signature
          differs, this will fail and we return False.
        """
        try:
            from ray import tune  # type: ignore

            tune.report(**metrics, checkpoint=checkpoint)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------
    def on_checkpoint(self, trainer: Any, path: str) -> bool:
        """
        Called by the Trainer after a checkpoint is saved.

        Parameters
        ----------
        trainer : Any
            Trainer-like object (unused here; included for hook signature consistency).
        path : str
            Directory path of the saved checkpoint.

        Returns
        -------
        bool
            Always True. This callback never requests early stopping.

        Notes
        -----
        Behavior:
          1) Convert checkpoint directory -> Ray AIR Checkpoint (if possible)
          2) Report to Ray AIR first
          3) If AIR reporting fails, fall back to legacy Tune reporting
          4) Ignore all failures (no-op if Ray is not available)
        """
        if self.report_every_saves <= 0:
            return True

        self._num_saves += 1
        if (self._num_saves % self.report_every_saves) != 0:
            return True

        ckpt = self._checkpoint_from_directory(path)
        if ckpt is None:
            return True

        metrics: Dict[str, Any] = {}
        if self.include_steps:
            self._maybe_add_steps(trainer, metrics)
        if self.metric_key:
            eval_metrics = getattr(trainer, "_last_eval_metrics", None)
            if not isinstance(eval_metrics, dict):
                eval_metrics = getattr(trainer, "last_eval_metrics", None)
            if isinstance(eval_metrics, dict) and (self.metric_key in eval_metrics):
                metrics[str(self.metric_key)] = eval_metrics[self.metric_key]

        if not metrics and not self.report_empty_metrics:
            return True

        # Prefer Ray AIR reporting.
        if self._report_air(metrics, ckpt):
            return True

        # Fallback: legacy Ray Tune reporting (best-effort).
        self._report_legacy(metrics, ckpt)
        return True
