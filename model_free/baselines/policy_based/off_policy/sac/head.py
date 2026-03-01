"""SAC network head definitions.

This module contains:

- :class:`SACHead`: policy/value network container for continuous-action SAC
- :func:`build_sac_head_worker_policy`: Ray-compatible worker reconstruction helper

The head is responsible for model construction, inference-oriented helpers, and
serialization-friendly metadata. Gradient update logic is delegated to
``SACCore``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.value_networks import DoubleStateActionValueNetwork
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)
from rllib.model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead


# =============================================================================
# Ray worker factory (module-level)
# =============================================================================
def build_sac_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
   Build a :class:`SACHead` instance on a Ray rollout worker (CPU-only).

    Ray reconstructs policies in remote worker processes using a serialized
    "factory spec" (entrypoint + kwargs). The entrypoint must be a module-level
    function so it is importable by name in a fresh Python process.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to :class:`SACHead`. These kwargs are typically
        JSON-serializable, coming from a :class:`~model_free.common.utils.ray_utils.PolicyFactorySpec`.
        The following adjustments are applied before construction:

        - ``device`` is forced to ``"cpu"`` to avoid accidental GPU allocation on
          rollout workers.
        - ``activation_fn`` may be a string identifier; it is resolved into an
          ``nn.Module`` class via :func:`~model_free.common.utils.ray_utils._resolve_activation_fn`.

    Returns
    -------
    torch.nn.Module
        A CPU-allocated :class:`SACHead` instance with ``set_training(False)``
        applied (best-effort) for inference-style rollout usage.

    Notes
    -----
    - Rollout workers generally should not run with dropout/batch-norm in training
      mode. This factory enforces eval-like behavior by calling
      ``head.set_training(False)``.
    - This function is intentionally defined at module scope (not inside a class)
      to satisfy Ray's entrypoint import rules.
    """
    kwargs = dict(kwargs)

    # Force CPU on rollout workers (avoid GPU contention and accidental allocations).
    kwargs["device"] = "cpu"

    # Resolve activation spec (e.g., "relu") -> nn.ReLU for constructor compatibility.
    kwargs["activation_fn"] = _resolve_activation_fn(kwargs.get("activation_fn", None))
    kwargs["feature_extractor_cls"] = _resolve_feature_extractor_cls(
        kwargs.get("feature_extractor_cls", None)
    )

    head = SACHead(**kwargs).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


# =============================================================================
# SACHead
# =============================================================================
class SACHead(OffPolicyContinuousActorCriticHead):
    """
   Soft Actor-Critic (SAC) head: stochastic actor + twin critics + target critics.

    This module bundles the neural networks that constitute SAC:

    - **Actor**: a squashed Gaussian policy :math:`\\pi_\\theta(a\\mid s)` that can
      sample actions and compute log-probabilities.
    - **Critic**: twin Q-functions :math:`Q_{\\phi_1}(s,a)`, :math:`Q_{\\phi_2}(s,a)`
      to mitigate positive bias (Double Q).
    - **Target critic**: a delayed (Polyak / hard-updated) copy used to form stable
      TD targets.

    The behavior/update logic is mostly implemented in the base class
    :class:`~model_free.common.policies.base_head.OffPolicyContinuousActorCriticHead`,
    which is expected to provide utilities such as:

    - device management and input conversion (e.g., ``_to_tensor_batched``)
    - action sampling + log-prob (SAC-style)
    - critic evaluation wrappers (e.g., ``q_values``, ``q_values_target``)
    - target network helpers (hard/soft update and freezing)

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension (continuous actions).
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for the actor and critic MLPs.
    activation_fn : Any, default=torch.nn.ReLU
        Activation function class (e.g., ``nn.ReLU``). This is stored as-is and
        passed to your network builders.
    init_type : str, default="orthogonal"
        Initialization scheme identifier forwarded to network modules.
    gain : float, default=1.0
        Initialization gain multiplier (network-implementation dependent).
    bias : float, default=0.0
        Initialization bias constant (network-implementation dependent).
    device : Union[str, torch.device], default=("cuda" if available else "cpu")
        Compute device for online and target networks.
    log_std_mode : str, default="layer"
        Policy log-standard-deviation parameterization mode (e.g., ``"layer"``,
        ``"parameter"``).
    log_std_init : float, default=-0.5
        Initial value for log standard deviation (implementation-dependent).

    Attributes
    ----------
    actor : torch.nn.Module
        Squashed Gaussian policy network.
    critic : torch.nn.Module
        Twin Q network producing :math:`(Q_1, Q_2)`.
    critic_target : torch.nn.Module
        Frozen target critic network for stable TD targets.

    Notes
    -----
    - Target critic weights are initialized by a hard copy from ``critic`` and then
      frozen to prevent optimizer updates and accidental gradient flow.
    - This head is designed to be serialized with JSON-safe kwargs and re-created
      on Ray workers via :meth:`get_ray_policy_factory_spec`.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
    ) -> None:
        """Initialize SAC actor, twin critics, and target critics.

        Parameters
        ----------
        obs_dim : int
            Flattened observation dimension.
        action_dim : int
            Continuous action dimension.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden sizes used by actor and critic backbones.
        activation_fn : Any, default=nn.ReLU
            Activation constructor for MLP blocks.
        init_type : str, default="orthogonal"
            Initialization strategy identifier.
        gain : float, default=1.0
            Initialization gain.
        bias : float, default=0.0
            Initialization bias value.
        obs_shape : tuple[int, ...], optional
            Optional raw observation shape for feature extractor selection.
        feature_extractor_cls : type[nn.Module], optional
            Optional custom feature extractor class.
        feature_extractor_kwargs : dict[str, Any], optional
            Keyword arguments for feature extractor construction.
        init_trunk : bool, optional
            Optional trunk-initialization override used by network builders.
        device : Union[str, torch.device], default=("cuda" if available else "cpu")
            Device on which modules are allocated.
        log_std_mode : str, default="layer"
            Log-standard-deviation parameterization mode for stochastic policy.
        log_std_init : float, default=-0.5
            Initial log-standard-deviation value.

        Returns
        -------
        None
            Constructor initializes module attributes in-place.

        Raises
        ------
        ValueError
            If ``obs_dim``, ``action_dim``, or any hidden size is not positive.
        """
        super().__init__(device=device)

        # ---------------------------------------------------------------------
        # Store configuration (useful for checkpoint metadata / Ray reconstruction)
        # ---------------------------------------------------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if len(self.hidden_sizes) == 0 or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be non-empty positive ints, got {self.hidden_sizes}")

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk

        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # ---------------------------------------------------------------------
        # Actor: squashed Gaussian policy (SAC-style)
        # ---------------------------------------------------------------------
        fe_a, fd_a = build_feature_extractor(
            obs_dim=self.obs_dim,
            obs_shape=self.obs_shape,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
        )
        self.actor = ContinuousPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=list(self.hidden_sizes),
            activation_fn=self.activation_fn,
            squash=True,
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
        # Critic: twin Q networks (Q1, Q2)
        # ---------------------------------------------------------------------
        self.critic = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            obs_shape=self.obs_shape,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Target critic: frozen copy of the critic used for TD targets
        # ---------------------------------------------------------------------
        self.critic_target = DoubleStateActionValueNetwork(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor_cls=self.feature_extractor_cls,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            obs_shape=self.obs_shape,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Initialize target weights from online critic, then freeze.
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Serialization helpers
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
       Export constructor kwargs in a JSON-safe form.

        This is used for:
        - checkpoint metadata (reproducibility / inspection)
        - Ray worker reconstruction (kwargs are serialized across processes)

        Returns
        -------
        Dict[str, Any]
            JSON-serializable constructor arguments. In particular:
            - ``activation_fn`` is converted into a stable string identifier.
            - ``device`` is stored as a string.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "activation_fn": self._activation_to_name(self.activation_fn),
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": (
                self.feature_extractor_cls
                if isinstance(self.feature_extractor_cls, str)
                else (
                    getattr(self.feature_extractor_cls, "__name__", None)
                    if self.feature_extractor_cls is not None
                    else None
                )
            ),
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "device": str(self.device),
            "log_std_mode": str(self.log_std_mode),
            "log_std_init": float(self.log_std_init),
        }

    def save(self, path: str) -> None:
        """
       Save actor/critic/target weights and minimal metadata to disk.

        Parameters
        ----------
        path : str
            Output path. If it does not end with ``".pt"``, the suffix is appended.

        Notes
        -----
        The serialized payload is a ``dict`` compatible with ``torch.save``:

        - ``"kwargs"``: JSON-safe constructor kwargs (for reconstruction / reference)
        - ``"actor"``: actor ``state_dict``
        - ``"critic"``: critic ``state_dict``
        - ``"critic_target"``: target critic ``state_dict``
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
       Load actor/critic/target weights from a checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path. If it does not end with ``".pt"``, the suffix is appended.

        Raises
        ------
        ValueError
            If the checkpoint payload does not match the expected format.

        Notes
        -----
        - Weights are loaded onto ``self.device`` via ``map_location``.
        - If ``critic_target`` weights are absent, the target is rebuilt by hard-copying
          the online critic to maintain backward compatibility with older checkpoints.
        - The target critic is re-frozen after loading and placed in eval mode.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized SACHead checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.freeze_target(self.critic_target)
        else:
            self.hard_update(self.critic_target, self.critic)
            self.freeze_target(self.critic_target)

        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
       Return a Ray-friendly factory spec for reconstructing this head on workers.

        Returns
        -------
        PolicyFactorySpec
            Factory specification containing:
            - ``entrypoint``: importable module-level factory function
            - ``kwargs``: JSON-safe constructor args

        Notes
        -----
        Ray workers typically run inference only, and the worker-side factory forces
        ``device="cpu"`` regardless of the stored device.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_sac_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
