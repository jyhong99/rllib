"""TQC network head.

This module defines :class:`TQCHead`, the model container for:

- squashed Gaussian actor,
- quantile critic ensemble,
- target quantile critic,
- serialization and Ray worker reconstruction helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn

from rllib.model_free.common.networks.policy_networks import ContinuousPolicyNetwork
from rllib.model_free.common.networks.feature_extractors import build_feature_extractor
from rllib.model_free.common.networks.value_networks import QuantileStateActionValueNetwork
from rllib.model_free.common.policies.base_head import OffPolicyContinuousActorCriticHead
from rllib.model_free.common.utils.ray_utils import (
    PolicyFactorySpec,
    _make_entrypoint,
    _resolve_activation_fn,
    _resolve_feature_extractor_cls,
)

# =============================================================================
# Ray worker factory (must be module-level for entrypoint resolution)
# =============================================================================
def build_tqc_head_worker_policy(**kwargs: Any) -> nn.Module:
    """
    Build a :class:`TQCHead` instance for Ray rollout workers (CPU-only).

    Ray integration requires policies to be reconstructible from an
    ``(entrypoint, kwargs)`` specification, where:

    - ``entrypoint`` is an importable module-level function (pickle/CloudPickle safe).
    - ``kwargs`` are JSON-serializable primitives (no classes, no devices, no tensors).

    This function enforces rollout-worker conventions:

    - **CPU forcing**: sets ``device="cpu"`` regardless of incoming configuration.
      This prevents accidental CUDA context creation and GPU contention in workers.
    - **Activation resolution**: converts ``activation_fn`` from a serialized
      representation (typically a string) into a ``torch.nn`` activation class.
    - **Eval-like mode**: calls ``head.set_training(False)`` so rollout behavior is
      deterministic with respect to dropout/batchnorm (if any) and target nets remain eval.

    Parameters
    ----------
    **kwargs : Any
        JSON-safe keyword arguments that match :class:`TQCHead` constructor inputs.
        In practice, these come from :meth:`TQCHead.get_ray_policy_factory_spec`.

    Returns
    -------
    torch.nn.Module
        A CPU-resident :class:`TQCHead` instance ready for inference/rollout.
    """
    cfg = dict(kwargs)
    cfg["device"] = "cpu"
    cfg["activation_fn"] = _resolve_activation_fn(cfg.get("activation_fn", None))
    cfg["feature_extractor_cls"] = _resolve_feature_extractor_cls(
        cfg.get("feature_extractor_cls", None)
    )

    head = TQCHead(**cfg).to("cpu")
    set_training_fn = getattr(head, "set_training", None)
    if callable(set_training_fn):
        set_training_fn(False)
    return head


# =============================================================================
# TQCHead (config-free)
# =============================================================================
class TQCHead(OffPolicyContinuousActorCriticHead):
    """
    Truncated Quantile Critics (TQC) head: actor + distributional critic ensemble + targets.

    This "head" owns the neural networks and inference utilities required by TQC:

    - **Actor**: a squashed Gaussian policy (SAC-style) producing actions and log-probs.
    - **Critic**: a *quantile* critic ensemble outputting a distribution over returns,
      represented as quantiles.
    - **Target critic**: a lagged copy of the critic used to build stable TD targets.

    The head is *network-centric* (construction + inference). Optimization and update
    logic (losses, truncation, temperature, Polyak updates) belongs in the *core*.

    Contract (expected by OffPolicyAlgorithm / core)
    -----------------------------------------------
    Required attributes/methods used downstream:

    - ``device`` : torch.device
    - ``set_training(training: bool)`` : toggle train/eval for online nets
    - ``act(obs, deterministic=False)`` : action selection (provided by base head)
    - ``sample_action_and_logp(obs)`` : stochastic sampling + log-prob (base head)
    - ``quantiles(obs, action) -> Tensor`` : quantiles from online critic
    - ``quantiles_target(obs, action) -> Tensor`` : quantiles from target critic
    - Target helpers: ``hard_update(...)``, ``soft_update(...)``, ``freeze_target(...)``
    - Persistence: ``save(path)``, ``load(path)``
    - Ray: ``get_ray_policy_factory_spec()``

    Shapes
    ------
    Let:
      - ``B`` be batch size
      - ``C`` be number of critic ensemble members (``n_nets``)
      - ``N`` be number of quantiles per critic (``n_quantiles``)

    Then quantile tensors have shape ``(B, C, N)``.

    Notes
    -----
    - This head does *not* implement truncation logic itself; truncation is typically
      applied in the core when computing targets/critic losses (TQC key idea).
    - Target critic parameters are frozen to avoid optimizer updates and accidental grads.
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
        # Actor distribution configuration (SAC-like)
        log_std_mode: str = "layer",
        log_std_init: float = -0.5,
        # Distributional critic configuration
        n_quantiles: int = 25,
        n_nets: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Observation vector dimension.
        action_dim : int
            Action vector dimension (continuous control).
        hidden_sizes : Sequence[int], default=(256, 256)
            MLP hidden layer widths used by both actor and critic networks.
        activation_fn : Any, default=torch.nn.ReLU
            Activation *class* used in MLP blocks (e.g., ``nn.ReLU``).
            For Ray, this is exported as a string and resolved back on workers.
        init_type : str, default="orthogonal"
            Initialization scheme name forwarded to network constructors.
        gain : float, default=1.0
            Gain forwarded to the initialization logic (commonly orthogonal gain).
        bias : float, default=0.0
            Bias initialization constant forwarded to network constructors.
        device : Union[str, torch.device], default="cuda" if available else "cpu"
            Device for learner-side networks. Ray workers override this to CPU.
        log_std_mode : str, default="layer"
            Policy log-std parameterization mode (depends on your actor implementation).
            Common options are "layer" (predict log-std from a layer) or "parameter"
            (global learnable log-std).
        log_std_init : float, default=-0.5
            Initial value for log standard deviation.
        n_quantiles : int, default=25
            Number of quantiles per critic head (``N``). Must be positive.
        n_nets : int, default=2
            Number of critic ensemble members (``C``). Must be positive.

        Raises
        ------
        ValueError
            If ``n_quantiles <= 0`` or ``n_nets <= 0``.
        """
        super().__init__(device=device)

        # -----------------------------
        # Core configuration
        # -----------------------------
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)

        self.activation_fn = activation_fn
        self.init_type = str(init_type)
        self.gain = float(gain)
        self.bias = float(bias)
        self.feature_extractor_cls = feature_extractor_cls
        self.feature_extractor_kwargs = dict(feature_extractor_kwargs or {})
        self.obs_shape = obs_shape
        self.init_trunk = init_trunk

        # Actor distribution hyperparameters
        self.log_std_mode = str(log_std_mode)
        self.log_std_init = float(log_std_init)

        # Quantile critic hyperparameters
        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)

        if self.n_quantiles <= 0:
            raise ValueError(f"n_quantiles must be positive, got {self.n_quantiles}")
        if self.n_nets <= 0:
            raise ValueError(f"n_nets must be positive, got {self.n_nets}")

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

        def _make_critic() -> nn.Module:
            return QuantileStateActionValueNetwork(
                state_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_quantiles=self.n_quantiles,
                n_nets=self.n_nets,
                hidden_sizes=self.hidden_sizes,
                activation_fn=self.activation_fn,
                feature_extractor_cls=self.feature_extractor_cls,
                feature_extractor_kwargs=self.feature_extractor_kwargs,
                obs_shape=self.obs_shape,
                init_type=self.init_type,
                gain=self.gain,
                bias=self.bias,
            ).to(self.device)

        self.critic = _make_critic()
        self.critic_target = _make_critic()

        # Initialize targets from online critic, then freeze to block gradients/optimizer.
        self.hard_update(self.critic_target, self.critic)
        self.freeze_target(self.critic_target)

    # =============================================================================
    # Quantile interfaces (TQC-specific)
    # =============================================================================
    def quantiles(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute online critic quantiles :math:`Z(s,a)`.

        Parameters
        ----------
        obs : Any
            Observations convertible by the base helper ``_to_tensor_batched``.
            Typical inputs include numpy arrays, torch tensors, lists, or single
            observations. Output will be batched as ``(B, obs_dim)``.
        action : Any
            Actions convertible by ``_to_tensor_batched``. Output will be batched
            as ``(B, action_dim)``.

        Returns
        -------
        torch.Tensor
            Quantile tensor with shape ``(B, C, N)`` where:
            ``C = n_nets`` and ``N = n_quantiles``.
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic(s, a)

    @th.no_grad()
    def quantiles_target(self, obs: Any, action: Any) -> th.Tensor:
        """
        Compute target critic quantiles :math:`Z_t(s,a)` (no gradients).

        Parameters
        ----------
        obs : Any
            Observation batch.
        action : Any
            Action batch.

        Returns
        -------
        torch.Tensor
            Target quantile tensor with shape ``(B, C, N)``.
        """
        s = self._to_tensor_batched(obs)
        a = self._to_tensor_batched(action)
        return self.critic_target(s, a)

    # =============================================================================
    # Persistence
    # =============================================================================
    def _export_kwargs_json_safe(self) -> Dict[str, Any]:
        """
        Export constructor kwargs in a JSON-safe format.

        This metadata is used for:
        - checkpoint reproducibility (store configuration alongside weights)
        - Ray worker reconstruction (kwargs must be JSON-serializable)

        Returns
        -------
        Dict[str, Any]
            JSON-safe constructor arguments. ``activation_fn`` is exported as a
            string name, and ``device`` is stringified.
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
            "n_quantiles": int(self.n_quantiles),
            "n_nets": int(self.n_nets),
        }

    def save(self, path: str) -> None:
        """
        Save actor + critic + target critic weights to a single checkpoint file.

        Parameters
        ----------
        path : str
            Checkpoint file path prefix. If ``path`` does not end with ``".pt"``,
            the extension is appended automatically.

        Notes
        -----
        The saved payload has the structure::

            {
              "kwargs": {...json-safe...},
              "actor": actor_state_dict,
              "critic": critic_state_dict,
              "critic_target": critic_target_state_dict,
            }
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
        Load actor + critic (+ optional target critic) weights from a checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint file path prefix. If ``path`` does not end with ``".pt"``,
            the extension is appended automatically.

        Raises
        ------
        ValueError
            If the checkpoint does not match the expected format.

        Notes
        -----
        - If ``critic_target`` is missing (backward compatibility), the target is
          rebuilt by hard-copying from the online critic.
        - Target critic parameters are always re-frozen after loading.
        """
        if not path.endswith(".pt"):
            path = path + ".pt"

        ckpt = th.load(path, map_location=self.device)
        if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
            raise ValueError(f"Unrecognized TQCHead checkpoint format at: {path}")

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        if ckpt.get("critic_target", None) is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.hard_update(self.critic_target, self.critic)

        self.freeze_target(self.critic_target)
        self.critic_target.eval()

    # =============================================================================
    # Ray integration
    # =============================================================================
    def get_ray_policy_factory_spec(self) -> PolicyFactorySpec:
        """
        Build a Ray-serializable factory spec for reconstructing this head on workers.

        Returns
        -------
        PolicyFactorySpec
            A spec containing:
            - ``entrypoint``: importable module-level factory function
            - ``kwargs``: JSON-safe constructor arguments
        """
        return PolicyFactorySpec(
            entrypoint=_make_entrypoint(build_tqc_head_worker_policy),
            kwargs=self._export_kwargs_json_safe(),
        )
