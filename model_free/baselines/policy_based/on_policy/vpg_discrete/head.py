"""Discrete VPG Head.

This module defines the discrete-action VPG network head and Ray worker
construction helper.

It provides:

- :func:`build_vpg_discrete_head_worker_policy` for JSON-safe worker rebuild.
- :class:`VPGDiscreteHead` with categorical actor, optional baseline critic,
  persistence helpers, and Ray factory-spec export.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.policy_networks import DiscretePolicyNetwork
from rllib.model_free.common.networks.value_networks import StateValueNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.policies.base_head import OnPolicyDiscreteActorCriticHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)

# =============================================================================
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_vpg_discrete_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Construct a :class:`VPGDiscreteHead` instance on a Ray worker (CPU-only).

    Ray commonly reconstructs policy modules inside remote worker processes using:
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
        Constructor keyword arguments intended for :class:`VPGDiscreteHead`.

        Serialization notes
        -------------------
        - This payload should be JSON/pickle-safe.
        - ``activation_fn`` is expected to be a serialized representation (string
          name or ``None``) and is resolved via :func:`_resolve_activation_fn`.

    Returns
    -------
    nn.Module
        A :class:`VPGDiscreteHead` allocated on CPU and set to inference mode via
        ``set_training(False)``.

    Notes
    -----
    - Rollout workers usually sample actions and do not perform gradient updates.
      Keeping them on CPU is typically simpler and more scalable.
    """
    # Defensive copy to avoid mutating caller state (important for Ray object reuse).
    cfg = dict(kwargs)

    # Force CPU on rollout workers.
    cfg["device"] = "cpu"

    # Resolve serialized activation function -> actual torch activation class/module.
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    # Construct the head and put it into inference mode.
    head = VPGDiscreteHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class VPGDiscreteHead(OnPolicyDiscreteActorCriticHead):
    """
    Vanilla Policy Gradient (VPG) head for discrete-action environments.

    This module defines the neural components needed by discrete-action VPG /
    REINFORCE-style methods:

    - **Actor**:
        Categorical policy :math:`\\pi(a\\mid s)` parameterized by logits produced
        by :class:`DiscretePolicyNetwork`.

    - **Critic** (optional baseline):
        State-value baseline :math:`V(s)` parameterized by :class:`StateValueNetwork`.
        The baseline is controlled by ``use_baseline`` and is used for variance
        reduction.

    Baseline modes
    --------------
    use_baseline=False
        REINFORCE-style training (no critic). Value baseline is not used.

    use_baseline=True
        Actor-critic VPG with a value baseline (recommended).

    Inherited API expectations
    --------------------------
    This class inherits from :class:`OnPolicyDiscreteActorCriticHead`, which is
    expected to provide:
    - ``act(obs, deterministic=False)``
    - ``evaluate_actions(obs, action, as_scalar=False)``
    - ``value_only(obs)``

    Notes
    -----
    - If your base head assumes a critic always exists, it must safely handle
      ``critic=None`` when ``use_baseline=False`` (e.g., return zeros for values).
      Your note indicates this compatibility is already implemented upstream.
    - This head exports JSON-safe constructor kwargs to support checkpoint metadata
      and Ray rollout-worker reconstruction.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
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
        use_baseline: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of the observation/state vector.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : Sequence[int], default=(64, 64)
            Hidden layer sizes for actor and critic MLPs.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function **class** (e.g., ``nn.ReLU``). For Ray and checkpoint
            metadata, this is typically serialized/deserialized via a stable name.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier used by the critic builder
            (the actor builder may ignore it depending on implementation).
        gain : float, default=1.0
            Optional initialization gain passed to the critic builder.
        bias : float, default=0.0
            Optional bias initialization constant passed to the critic builder.
        device : Union[str, torch.device], default="cpu"
            Device on which the head parameters are allocated.
        use_baseline : bool, default=True
            Whether to include a value baseline network :math:`V(s)`.

        Notes
        -----
        - This class defines networks only; learning rules live in the core.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)

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
        # Actor: categorical policy network for discrete actions
        # ---------------------------------------------------------------------
        # The DiscretePolicyNetwork is expected to expose get_dist(obs) returning
        # a Categorical-like distribution with:
        #   - sample()
        #   - log_prob(action)
        #   - entropy()
        fe_a, fd_a = _build_fe()
        self.actor = DiscretePolicyNetwork(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            feature_extractor=fe_a,
            feature_dim=fd_a,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Critic (optional): value baseline V(s)
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
    # Persistence / JSON-safe kwargs export
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor arguments as a JSON-safe dictionary.

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
            "n_actions": int(self.n_actions),
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
        - Optimizer state belongs to the core/algorithm, not the head.
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
            raise ValueError(
                "Unrecognized VPGDiscreteHead checkpoint payload (missing actor)."
            )

        # Always load actor weights.
        self.actor.load_state_dict(ckpt["actor"])

        ckpt_critic = ckpt.get("critic", None)

        # Baseline compatibility checks.
        if self.critic is None:
            # Current instance baseline OFF.
            if ckpt_critic is not None:
                raise ValueError(
                    "Checkpoint contains critic weights but this VPGDiscreteHead has use_baseline=False."
                )
        else:
            # Current instance baseline ON.
            if ckpt_critic is None:
                raise ValueError(
                    "Checkpoint has no critic weights but this VPGDiscreteHead has use_baseline=True."
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
            entrypoint=_make_entrypoint(build_vpg_discrete_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
