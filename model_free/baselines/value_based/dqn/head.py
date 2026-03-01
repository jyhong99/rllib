"""DQN Head.

This module defines the DQN network head and Ray worker reconstruction helper.

It provides:

- :func:`build_dqn_head_worker_policy` for CPU-only worker reconstruction.
- :class:`DQNHead` with online/target Q-networks, checkpoint helpers, and Ray
  factory-spec export.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.q_networks import QNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.policies.base_head import QLearningHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)

# =============================================================================
# Ray worker factory (MUST be module-level for Ray serialization)
# =============================================================================
def build_dqn_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Construct a :class:`DQNHead` instance on a Ray worker (CPU-only).

    Ray commonly reconstructs policies in remote worker processes using:
    - a **pickle-friendly entrypoint** (must be a module-level symbol), and
    - a **JSON/pickle-safe kwargs** payload.

    This factory enforces rollout-worker constraints:

    - **CPU-only instantiation**:
        Rollout workers typically do inference only (action selection) and should
        not compete for GPU resources.

    - **Activation resolution**:
        ``activation_fn`` is usually serialized as a string (e.g., ``"ReLU"``).
        This factory resolves it back into an actual activation class via
        :func:`_resolve_activation_fn`.

    - **Inference mode**:
        ``set_training(False)`` disables training-specific behavior (e.g., dropout)
        and establishes the convention that workers do not perform gradient updates.

    Parameters
    ----------
    **kwargs : Any
        JSON/pickle-safe keyword arguments intended for :class:`DQNHead`.

        Serialization notes
        -------------------
        - ``activation_fn`` is expected to be a serialized identifier (string/None).
        - ``device`` will be overridden to ``"cpu"`` regardless of the payload.

    Returns
    -------
    nn.Module
        A :class:`DQNHead` allocated on CPU and set to inference mode.

    Notes
    -----
    This function must be importable at module scope for Ray to pickle/deserialize
    the construction entrypoint.
    """
    cfg = dict(kwargs)

    # Rollout workers should not depend on GPU availability.
    cfg["device"] = "cpu"

    # activation_fn is serialized as string/None -> resolve to callable/class.
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(cfg.get("feature_extractor_cls", None))

    head = DQNHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


class DQNHead(QLearningHead):
    """
    DQN head: online Q-network + target Q-network for discrete control.

    This "head" is a thin network container used by DQN-style cores/algorithms:

    - **Online network** ``q``:
        Computes :math:`Q_\\theta(s, a)` for action selection and TD learning.

    - **Target network** ``q_target``:
        Computes :math:`Q_{\\bar\\theta}(s, a)` for bootstrapped TD targets.
        It is typically updated periodically (hard update) or softly (Polyak) by
        the core/algorithm.

    Contract (expected by DQN-style cores / off-policy drivers)
    -----------------------------------------------------------
    This class is designed to work with off-policy training loops that expect:

    - ``device`` : torch.device-like
    - ``set_training(training: bool) -> None``
    - ``act(obs, epsilon=0.0, deterministic=True) -> torch.Tensor`` (B,) long
    - ``q_values(obs) -> torch.Tensor`` (B, A)
    - ``q_values_target(obs) -> torch.Tensor`` (B, A)
    - ``save(path)``, ``load(path)``
    - ``get_ray_policy_factory_spec()``

    Some of these methods are typically provided by :class:`QLearningHead`.

    Design notes
    ------------
    - The target network is conventionally **frozen** (``requires_grad=False``) and
      placed into **eval** mode. This prevents accidental gradient flow through the
      target and avoids training-mode behavior (dropout/bn) if present.
    - This head initializes the target network by copying the online parameters via
      ``hard_update`` and then freezing it via ``freeze_target``.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation_fn: Any = nn.ReLU,
        dueling_mode: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        feature_extractor_cls: Optional[type[nn.Module]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        init_trunk: bool | None = None,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        device: Union[str, th.device] = "cuda" if th.cuda.is_available() else "cpu",
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimension of the observation/state vector.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : Sequence[int], default=(256, 256)
            Hidden layer sizes of the Q-network MLP.
        activation_fn : Any, default=torch.nn.ReLU
            Activation function **class** used by the Q-network builder.
            For Ray reconstruction, this is typically serialized as a name string.
        dueling_mode : bool, default=False
            If True, use a dueling architecture (separate value/advantage streams)
            inside :class:`QNetwork`.
        init_type : str, default="orthogonal"
            Weight initialization scheme identifier passed to :class:`QNetwork`.
        gain : float, default=1.0
            Optional initialization gain multiplier passed to :class:`QNetwork`.
        bias : float, default=0.0
            Optional bias initialization constant passed to :class:`QNetwork`.
        device : Union[str, torch.device], default="cuda" if available else "cpu"
            Device on which the head parameters are allocated.

        Notes
        -----
        - The base class :class:`QLearningHead` is expected to provide helper methods:
          ``hard_update(dst, src)`` and ``freeze_target(module)``.
        - The target network is created with the same architecture as the online
          network and then synchronized immediately.
        """
        super().__init__(device=device)

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation_fn = activation_fn
        self.dueling_mode = bool(dueling_mode)
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk

        def _build_fe() -> Tuple[Optional[nn.Module], Optional[int]]:
            """Build one feature-extractor pair for a Q-network branch."""
            return build_feature_extractor(
                obs_dim=self.obs_dim,
                obs_shape=self.obs_shape,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
            )

        # ---------------------------------------------------------------------
        # Online Q-network: Q(s, ·) for action selection and TD learning
        # ---------------------------------------------------------------------
        fe_q, fd_q = _build_fe()
        self.q = QNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            feature_extractor=fe_q,
            feature_dim=fd_q,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Target Q-network: Q_target(s, ·) for bootstrap targets
        # ---------------------------------------------------------------------
        fe_t, fd_t = _build_fe()
        self.q_target = QNetwork(
            state_dim=self.obs_dim,
            action_dim=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            activation_fn=self.activation_fn,
            dueling_mode=self.dueling_mode,
            feature_extractor=fe_t,
            feature_dim=fd_t,
            init_trunk=self.init_trunk,
            init_type=self.init_type,
            gain=self.gain,
            bias=self.bias,
        ).to(self.device)

        # Sync target params = online params (hard update), then freeze target.
        self.hard_update(self.q_target, self.q)
        self.freeze_target(self.q_target)

    # =============================================================================
    # Persistence / JSON-safe kwargs export
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-serializable form.

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor kwargs suitable for:
            - checkpoint metadata (reconstruction/debugging)
            - Ray worker instantiation (kwargs must be serializable)

        Notes
        -----
        - ``activation_fn`` is converted to a string because function/class objects
          are not JSON-serializable. The worker resolves it back via
          :func:`_resolve_activation_fn`.
        - ``device`` is included for transparency; Ray workers override it to CPU.
        """
        act_name = self._activation_to_name(self.activation_fn)
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
            "activation_fn": act_name,
            "dueling_mode": bool(self.dueling_mode),
            "obs_shape": tuple(self.obs_shape) if self.obs_shape is not None else None,
            "feature_extractor_cls": fe_name,
            "feature_extractor_kwargs": dict(self.feature_extractor_kwargs or {}),
            "init_trunk": self.init_trunk,
            "init_type": str(self.init_type),
            "gain": float(self.gain),
            "bias": float(self.bias),
            "device": str(self.device),
        }

    def save(self, path: str) -> None:
        """
        Save a DQNHead checkpoint to disk.

        Parameters
        ----------
        path : str
            Output path. The suffix ``.pt`` is appended if missing.

        Stored payload
        --------------
        kwargs : dict
            JSON-safe constructor kwargs (for reproducibility/reconstruction).
        q : dict
            Online network ``state_dict``.
        q_target : dict
            Target network ``state_dict``.

        Notes
        -----
        This is a head-only checkpoint. Optimizer state belongs to the core/algorithm
        and is intentionally not stored here.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        payload: Dict[str, Any] = {
            "kwargs": self._export_kwargs_json_safe(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        th.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a checkpoint produced by :meth:`save`. The suffix ``.pt`` is
            appended if missing.

        Raises
        ------
        ValueError
            If the checkpoint payload is not recognized.

        Notes
        -----
        - Loads onto ``self.device`` via ``map_location``.
        - Re-applies the frozen/eval invariant for the target net after loading.
        - If ``q_target`` is missing (older checkpoints), falls back to syncing the
          target network from the online network.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q" not in ckpt:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        self.q.load_state_dict(ckpt["q"])

        if "q_target" in ckpt and ckpt["q_target"] is not None:
            self.q_target.load_state_dict(ckpt["q_target"])
            self.freeze_target(self.q_target)
        else:
            self.hard_update(self.q_target, self.q)
            self.freeze_target(self.q_target)

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Return a Ray-friendly construction spec for this head.

        Returns
        -------
        PolicyFactorySpec
            Spec containing:
            - ``entrypoint`` : module-level worker factory (pickle-friendly)
            - ``kwargs``     : JSON-safe constructor args (portable across workers)

        Notes
        -----
        Worker policies are constructed by calling::

            head = build_dqn_head_worker_policy(**kwargs)

        The worker factory overrides device placement to CPU.
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_dqn_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
