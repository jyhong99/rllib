"""VPG Head.

This module defines the continuous-action VPG network head and Ray worker
construction helper.

It provides:

- :func:`build_vpg_head_worker_policy` for JSON-safe worker reconstruction.
- :class:`VPGHead` with Gaussian actor, optional baseline critic, persistence
  helpers, and Ray factory-spec export.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from rllib.model_free.common.networks.value_networks import StateValueNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.policies.base_head import OnPolicyContinuousActorCriticHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)

# =============================================================================
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_vpg_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Construct a :class:`VPGHead` instance on a Ray worker (CPU-only).

    Ray typically reconstructs policy modules inside remote worker processes using:
    - a **pickleable entrypoint** (must be a module-level symbol), and
    - a **JSON/pickle-safe kwargs payload**.

    This factory enforces rollout-worker constraints:
    - **CPU-only** instantiation to avoid GPU contention and reduce overhead,
    - **activation resolution** from serialized identifiers (e.g., "ReLU") back
      to a torch activation class,
    - **inference mode** by disabling training-specific behaviors.

    Parameters
    ----------
    **kwargs : Any
        Constructor keyword arguments intended for :class:`VPGHead`.

        Serialization notes
        -------------------
        - This payload should be JSON/pickle-safe.
        - ``activation_fn`` is expected to be a serialized representation (string
          name or ``None``) and is resolved via :func:`_resolve_activation_fn`.

    Returns
    -------
    nn.Module
        A :class:`VPGHead` allocated on CPU and set to inference mode via
        ``set_training(False)``.

    Notes
    -----
    - Rollout workers usually sample actions and do not perform gradient updates.
      Keeping them on CPU is typically simpler and more scalable.
    """
    # Defensive copy to avoid mutating caller state (important for Ray object reuse).
    cfg = dict(kwargs)

    # Force CPU for worker-side policy instantiation.
    cfg["device"] = "cpu"

    # Convert serialized activation function name/string -> torch module/class.
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    # Build head and ensure it is in inference mode on workers.
    head = VPGHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class VPGHead(OnPolicyContinuousActorCriticHead):
    """
    Vanilla Policy Gradient (VPG) head for continuous actions.

    This module provides the neural components for policy-gradient style methods:

    - **Actor**: diagonal Gaussian policy :math:`\\pi(a\\mid s)` (unsquashed)
    - **Critic** (optional): state-value baseline :math:`V(s)` for variance reduction

    Modes
    -----
    use_baseline=False
        REINFORCE-style training (no critic). Advantages are typically computed
        directly from returns (A = R) or by upstream logic that does not require
        value predictions.

    use_baseline=True
        VPG with a value baseline (recommended). Advantages may be computed as
        A = R - V(s), or by upstream GAE logic that uses V(s).

    API assumptions
    ---------------
    This class inherits from :class:`OnPolicyContinuousActorCriticHead`, which is
    expected to provide common inference/training utilities such as:
    - ``act(obs, deterministic=False)``
    - ``evaluate_actions(obs, act)``
    - ``value_only(obs)``

    Notes
    -----
    - The actor uses an **unsquashed** Gaussian policy (``squash=False``), which
      is the simplest continuous-control formulation. If you squash actions
      (e.g., tanh), you must handle the induced distribution transform and
      log-prob corrections.
    - For Ray integration, this head exports JSON-safe constructor kwargs and
      provides a module-level worker factory.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation_fn: Any = nn.ReLU,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cpu",
        # Gaussian policy parameters
        log_std_mode: str = "param",
        log_std_init: float = -0.5,
        # Baseline toggle
        use_baseline: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of the observation/state vector.
        action_dim : int
            Dimension of the continuous action vector.
        hidden_sizes : Sequence[int], default=(64, 64)
            Hidden layer sizes for the actor MLP (and critic MLP if enabled).
        activation_fn : Any, default=torch.nn.ReLU
            Activation function **class** (e.g., ``nn.ReLU``). For Ray and checkpoint
            metadata, this is typically serialized/deserialized via a stable name.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier (interpreted by your network builders).
        gain : float, default=1.0
            Optional initialization gain passed to network builders.
        bias : float, default=0.0
            Optional bias initialization constant passed to network builders.
        device : Union[str, torch.device], default="cpu"
            Device on which the head parameters are allocated.
        log_std_mode : str, default="param"
            Gaussian log-standard-deviation parameterization mode for the actor.
        log_std_init : float, default=-0.5
            Initial value for the Gaussian log-std parameters when applicable.
        use_baseline : bool, default=True
            Whether to build a value-network baseline :math:`V(s)`.

        Notes
        -----
        - This class does not implement the learning rule; it only defines networks.
        - If ``use_baseline=False``, downstream code must not assume critic/value
          outputs are present.
        """
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Core dimensions / configuration
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        self.use_baseline = bool(use_baseline)

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build a feature extractor pair for one network branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        # ---------------------------------------------------------------------
        # Actor: unsquashed diagonal Gaussian policy π(a|s)
        # ---------------------------------------------------------------------
        fe_a, fd_a = _build_fe()
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=False,
            log_std_mode=self.log_std_mode,
            log_std_init=self.log_std_init,
            feature_extractor=fe_a,
            feature_dim=fd_a,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Optional critic: state-value baseline V(s)
        # ---------------------------------------------------------------------
        self.critic: Optional[StateValueNetwork] = None
        if self.use_baseline:
            fe_c, fd_c = _build_fe()
            self.critic = StateValueNetwork(
                state_dim=self.obs_dim,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                feature_extractor=fe_c,
                feature_dim=fd_c,
                init_trunk=self.init_trunk,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

    # =============================================================================
    # Persistence + Ray reconstruction kwargs
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments in a JSON-safe form.

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor kwargs suitable for:
            - checkpoint metadata (reconstruction/debugging)
            - Ray worker instantiation (kwargs must be serializable)

        Notes
        -----
        - ``activation_fn`` is converted to a stable string identifier using
          ``_activation_to_name`` (provided by the base head).
        - ``device`` is included for transparency; Ray workers override it to "cpu".
        """
        fe_cls = self.feature_extractor_cls
        if isinstance(fe_cls, str):
            fe_name: Optional[str] = fe_cls
        elif fe_cls is not None:
            fe_name = getattr(fe_cls, "__name__", None) or str(fe_cls)
        else:
            fe_name = None

        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": fe_name,
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
            "log_std_mode": str(self.log_std_mode),
            "log_std_init": float(self.log_std_init),
            "use_baseline": bool(self.use_baseline),
        }

    def save(self, path: str) -> None:
        """
        Save head weights and metadata to disk.

        Parameters
        ----------
        path : str
            Output path. The suffix ``.pt`` is appended if missing.

        Stored payload
        --------------
        kwargs : dict
            JSON-safe config used to build this head (for reproducibility/debugging).
        actor : dict
            Actor ``state_dict``.
        critic : dict or None
            Critic ``state_dict`` if baseline is enabled, otherwise ``None``.

        Notes
        -----
        - Optimizer/scheduler state belongs to the core/algorithm, not the head.
        """
        if not path.endswith(".pt"):
            path += ".pt"

        th.save(
            {
                "kwargs": self._export_kwargs_json_safe(),
                "actor": self.actor.state_dict(),
                "critic": None if self.critic is None else self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load checkpoint weights into the existing instance.

        Parameters
        ----------
        path : str
            Path to checkpoint saved by :meth:`save`. The suffix ``.pt`` is appended
            if missing.

        Raises
        ------
        ValueError
            If the checkpoint payload is not recognized or baseline compatibility fails.

        Notes
        -----
        - Loads weights only; does not reconstruct a new object.
        - Uses ``map_location=self.device`` for CPU/GPU portability.
        - Enforces baseline compatibility:
            * baseline OFF instance cannot load baseline ON checkpoint (critic mismatch)
            * baseline ON  instance cannot load baseline OFF checkpoint
        """
        if not path.endswith(".pt"):
            path += ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognized checkpoint format: {path}")

        if "actor" not in ckpt:
            raise ValueError("Unrecognized VPGHead checkpoint payload (missing actor).")

        # Always load actor weights.
        self.actor.load_state_dict(ckpt["actor"])

        ckpt_critic = ckpt.get("critic", None)

        # Baseline compatibility checks.
        if self.critic is None:
            # Current instance baseline OFF.
            if ckpt_critic is not None:
                raise ValueError(
                    "Checkpoint contains critic weights but this VPGHead has use_baseline=False."
                )
        else:
            # Current instance baseline ON.
            if ckpt_critic is None:
                raise ValueError(
                    "Checkpoint has no critic weights but this VPGHead has use_baseline=True."
                )
            self.critic.load_state_dict(ckpt_critic)

    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-safe construction spec for this head.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint`` : module-level worker factory (pickle-friendly)
            - ``kwargs``     : JSON-safe constructor args (portable across workers)

        Notes
        -----
        - Ray requires the entrypoint to be a module-level symbol so it can be pickled.
        - Worker policies are typically CPU-only; the worker factory enforces that.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_vpg_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
