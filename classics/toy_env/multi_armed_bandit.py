from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore
        spaces = None  # type: ignore


@dataclass(frozen=True)
class BanditConfig:
    """
    Configuration for a Gaussian multi-armed bandit.

    Parameters
    ----------
    means : Tuple[float, ...]
        Mean reward for each arm.
    stds : Tuple[float, ...]
        Standard deviation of reward noise for each arm.
    episode_length : int, default=1
        Number of pulls per episode before termination.
    """

    means: Tuple[float, ...]
    stds: Tuple[float, ...]
    episode_length: int = 1


class MultiArmedBanditEnv((gym.Env if gym is not None else object)):  # type: ignore[misc]
    """
    Stateless Gaussian multi-armed bandit environment.

    Observation space is a constant scalar (shape ``(1,)``) and action space is
    ``Discrete(n_arms)``.

    Notes
    -----
    Rewards are sampled as ``r ~ Normal(means[action], stds[action])``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        means: Optional[Sequence[float]] = None,
        stds: Optional[Sequence[float]] = None,
        episode_length: int = 1,
    ) -> None:
        """
        Initialize a Gaussian multi-armed bandit.

        Parameters
        ----------
        means : Optional[Sequence[float]], default=None
            Mean reward per arm. If ``None``, a 10-arm default setting is used.
        stds : Optional[Sequence[float]], default=None
            Reward standard deviation per arm. If ``None``, all arms use ``1.0``.
        episode_length : int, default=1
            Number of interactions before ``terminated=True``.

        Raises
        ------
        ValueError
            If means are empty, stds length mismatches means, a std is negative,
            or episode length is not positive.
        ImportError
            If neither Gymnasium nor Gym is available.
        """
        if means is None:
            means = [0.2, 0.0, -0.2, 0.8, 0.4, -0.5, 0.1, 0.6, -0.1, 0.3]
        if stds is None:
            stds = [1.0] * len(means)

        if len(means) == 0:
            raise ValueError("means must be non-empty")
        if len(stds) != len(means):
            raise ValueError("stds must have the same length as means")
        if any(s < 0 for s in stds):
            raise ValueError("stds values must be >= 0")
        if int(episode_length) <= 0:
            raise ValueError("episode_length must be >= 1")

        self.config = BanditConfig(
            means=tuple(float(m) for m in means),
            stds=tuple(float(s) for s in stds),
            episode_length=int(episode_length),
        )

        if spaces is None:  # pragma: no cover
            raise ImportError("gymnasium or gym is required to use MultiArmedBanditEnv")

        self.action_space = spaces.Discrete(len(self.config.means))
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([0.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        self._obs = np.array([0.0], dtype=np.float32)
        self._rng = np.random.default_rng()
        self._step_count = 0

    @property
    def n_arms(self) -> int:
        """int: Number of bandit arms."""
        return self.action_space.n

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Reset RNG state.

        Parameters
        ----------
        seed : Optional[int], default=None
            Seed used to initialize NumPy's ``default_rng``.
        """
        self._rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset episode state.

        Parameters
        ----------
        seed : Optional[int], default=None
            Optional RNG seed applied before the episode starts.
        options : Optional[Dict[str, Any]], default=None
            Reserved for API compatibility; currently unused.

        Returns
        -------
        obs : np.ndarray
            Constant observation of shape ``(1,)``.
        info : Dict[str, Any]
            Bandit metadata, including arm means/stds and best arm statistics.
        """
        del options
        if seed is not None:
            self.seed(seed)

        self._step_count = 0
        info = {
            "means": np.asarray(self.config.means, dtype=np.float32),
            "stds": np.asarray(self.config.stds, dtype=np.float32),
            "best_arm": int(np.argmax(self.config.means)),
            "best_mean": float(np.max(self.config.means)),
        }
        return self._obs.copy(), info

    def step(self, action: int):
        """
        Pull one arm and sample reward.

        Parameters
        ----------
        action : int
            Arm index in ``[0, n_arms - 1]``.

        Returns
        -------
        obs : np.ndarray
            Constant observation of shape ``(1,)``.
        reward : float
            Sampled reward for the selected arm.
        terminated : bool
            ``True`` when episode length is reached.
        truncated : bool
            Always ``False`` in this environment.
        info : Dict[str, Any]
            Step diagnostics including selected arm and instantaneous regret.

        Raises
        ------
        ValueError
            If action is outside the valid arm range.
        """
        action = int(action)
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms - 1}], got {action}")

        mean = self.config.means[action]
        std = self.config.stds[action]
        reward = float(self._rng.normal(loc=mean, scale=std))

        self._step_count += 1
        terminated = bool(self._step_count >= self.config.episode_length)
        truncated = False
        info = {
            "arm": action,
            "arm_mean": float(mean),
            "arm_std": float(std),
            "regret": float(np.max(self.config.means) - mean),
            "step_count": self._step_count,
        }
        return self._obs.copy(), reward, terminated, truncated, info

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize environment state.

        Returns
        -------
        Dict[str, Any]
            Serializable state containing step counter, RNG state, and config.
        """
        return {
            "step_count": int(self._step_count),
            "rng_state": self._rng.bit_generator.state,
            "config": {
                "means": list(self.config.means),
                "stds": list(self.config.stds),
                "episode_length": self.config.episode_length,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore environment state.

        Parameters
        ----------
        state : Dict[str, Any]
            State produced by :meth:`state_dict`.
        """
        self._step_count = int(state.get("step_count", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state


def register_multi_armed_bandit_env(env_id: str = "MultiArmedBandit-v0") -> None:
    """
    Register ``MultiArmedBanditEnv`` with Gymnasium/Gym.

    Parameters
    ----------
    env_id : str, default=\"MultiArmedBandit-v0\"
        Registry id to associate with this environment class.
    """
    if gym is None:  # pragma: no cover
        raise ImportError("gymnasium or gym is required for env registration")

    try:
        registry = gym.envs.registry  # gymnasium and gym expose this
        if env_id in registry:
            return
    except Exception:
        pass

    gym.register(id=env_id, entry_point="classics.toy_env.multi_armed_bandit:MultiArmedBanditEnv")
