from __future__ import annotations
"""
Classic multi-armed and contextual bandit algorithms.

This module provides simple tabular implementations of several widely used
bandit methods and small runner utilities for Gym/Gymnasium-like environments.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np


class BanditAgent(Protocol):
    """
    Protocol for non-contextual bandit agents.

    Notes
    -----
    Agents implementing this protocol can be used with :func:`run_bandit`.
    """

    def select_action(self) -> int:
        ...

    def update(self, action: int, reward: float) -> None:
        ...


@dataclass
class BanditRunResult:
    """
    Collected trajectory and aggregate metrics for one run.

    Attributes
    ----------
    actions : np.ndarray
        Chosen arm indices, shape ``(n_steps,)``.
    rewards : np.ndarray
        Observed rewards, shape ``(n_steps,)``.
    mean_reward : float
        Mean reward over the run.
    cumulative_reward : float
        Sum of rewards over the run.
    """

    actions: np.ndarray
    rewards: np.ndarray
    mean_reward: float
    cumulative_reward: float


@dataclass
class ContextualBanditRunResult:
    """
    Collected trajectory and aggregate metrics for one contextual-bandit run.

    Attributes
    ----------
    actions : np.ndarray
        Chosen arm indices, shape ``(n_steps,)``.
    rewards : np.ndarray
        Observed rewards, shape ``(n_steps,)``.
    mean_reward : float
        Mean reward over the run.
    cumulative_reward : float
        Sum of rewards over the run.
    """

    actions: np.ndarray
    rewards: np.ndarray
    mean_reward: float
    cumulative_reward: float


class EpsilonGreedyAgent:
    """
    Epsilon-greedy incremental action-value bandit.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    epsilon : float, default=0.1
        Exploration probability.
    initial_value : float, default=0.0
        Initial action-value estimate for all arms.
    step_size : Optional[float], default=None
        Constant update step-size. If ``None``, uses sample-average updates.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        step_size: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if step_size is not None and step_size <= 0.0:
            raise ValueError("step_size must be > 0 when provided")

        self.n_arms = int(n_arms)
        self.epsilon = float(epsilon)
        self.step_size = step_size
        self.q_values = np.full(self.n_arms, float(initial_value), dtype=np.float64)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    def _argmax_tie_break(self, values: np.ndarray) -> int:
        max_value = float(np.max(values))
        candidates = np.flatnonzero(values == max_value)
        return int(self._rng.choice(candidates))

    def select_action(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return self._argmax_tie_break(self.q_values)

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        if self.step_size is None:
            alpha = 1.0 / float(self.counts[action])
        else:
            alpha = float(self.step_size)
        self.q_values[action] += alpha * (float(reward) - self.q_values[action])


class OptimisticInitialValuesAgent(EpsilonGreedyAgent):
    """
    Greedy agent with optimistic initialization.

    Notes
    -----
    Sets ``epsilon=0`` and uses large initial values to induce exploration.
    """

    def __init__(
        self,
        n_arms: int,
        optimistic_value: float = 5.0,
        step_size: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            epsilon=0.0,
            initial_value=optimistic_value,
            step_size=step_size,
            seed=seed,
        )


class UCBAgent:
    """
    Upper-Confidence-Bound (UCB1-style) bandit agent.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    c : float, default=2.0
        Exploration coefficient for confidence bonus.
    initial_value : float, default=0.0
        Initial value estimate per arm.
    step_size : Optional[float], default=None
        Constant step-size. If ``None``, uses sample-average updates.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        initial_value: float = 0.0,
        step_size: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if c < 0.0:
            raise ValueError("c must be >= 0")
        if step_size is not None and step_size <= 0.0:
            raise ValueError("step_size must be > 0 when provided")

        self.n_arms = int(n_arms)
        self.c = float(c)
        self.step_size = step_size
        self.q_values = np.full(self.n_arms, float(initial_value), dtype=np.float64)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.t = 0
        self._rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        self.t += 1
        untried = np.flatnonzero(self.counts == 0)
        if untried.size > 0:
            return int(self._rng.choice(untried))

        bonus = self.c * np.sqrt(np.log(float(self.t)) / self.counts.astype(np.float64))
        ucb_values = self.q_values + bonus
        max_value = float(np.max(ucb_values))
        candidates = np.flatnonzero(ucb_values == max_value)
        return int(self._rng.choice(candidates))

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        if self.step_size is None:
            alpha = 1.0 / float(self.counts[action])
        else:
            alpha = float(self.step_size)
        self.q_values[action] += alpha * (float(reward) - self.q_values[action])


class GradientBanditAgent:
    """
    Gradient bandit with softmax preferences.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    alpha : float, default=0.1
        Preference update step-size.
    use_baseline : bool, default=True
        Whether to subtract running average reward as baseline.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 0.1,
        use_baseline: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")

        self.n_arms = int(n_arms)
        self.alpha = float(alpha)
        self.use_baseline = bool(use_baseline)

        self.preferences = np.zeros(self.n_arms, dtype=np.float64)
        self.avg_reward = 0.0
        self.t = 0
        self._rng = np.random.default_rng(seed)

    def _policy(self) -> np.ndarray:
        shifted = self.preferences - float(np.max(self.preferences))
        probs = np.exp(shifted)
        probs /= np.sum(probs)
        return probs

    def select_action(self) -> int:
        probs = self._policy()
        return int(self._rng.choice(self.n_arms, p=probs))

    def update(self, action: int, reward: float) -> None:
        reward = float(reward)
        self.t += 1
        baseline = self.avg_reward if self.use_baseline else 0.0
        probs = self._policy()

        one_hot = np.zeros(self.n_arms, dtype=np.float64)
        one_hot[action] = 1.0
        self.preferences += self.alpha * (reward - baseline) * (one_hot - probs)

        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / float(self.t)


class ThompsonSamplingAgent:
    """
    Gaussian Thompson Sampling with Normal posteriors.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    prior_mean : float, default=0.0
        Prior mean per arm.
    prior_precision : float, default=1.0
        Prior precision (inverse variance) per arm.
    reward_precision : float, default=1.0
        Observation precision.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        reward_precision: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if prior_precision <= 0.0 or reward_precision <= 0.0:
            raise ValueError("prior_precision and reward_precision must be > 0")

        self.n_arms = int(n_arms)
        self.posterior_mean = np.full(self.n_arms, float(prior_mean), dtype=np.float64)
        self.posterior_precision = np.full(self.n_arms, float(prior_precision), dtype=np.float64)
        self.reward_precision = float(reward_precision)
        self._rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        std = 1.0 / np.sqrt(self.posterior_precision)
        samples = self._rng.normal(loc=self.posterior_mean, scale=std)
        max_value = float(np.max(samples))
        candidates = np.flatnonzero(samples == max_value)
        return int(self._rng.choice(candidates))

    def update(self, action: int, reward: float) -> None:
        prior_mean = self.posterior_mean[action]
        prior_precision = self.posterior_precision[action]
        post_precision = prior_precision + self.reward_precision
        post_mean = (prior_precision * prior_mean + self.reward_precision * float(reward)) / post_precision

        self.posterior_mean[action] = post_mean
        self.posterior_precision[action] = post_precision


class EXP3Agent:
    """
    EXP3 algorithm for adversarial bandits.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    gamma : float, default=0.07
        Exploration/mixing parameter.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(self, n_arms: int, gamma: float = 0.07, seed: Optional[int] = None) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.n_arms = int(n_arms)
        self.gamma = float(gamma)
        self.weights = np.ones(self.n_arms, dtype=np.float64)
        self._last_probs = np.full(self.n_arms, 1.0 / self.n_arms, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def _probs(self) -> np.ndarray:
        norm = self.weights / np.sum(self.weights)
        probs = (1.0 - self.gamma) * norm + self.gamma / float(self.n_arms)
        return probs

    def select_action(self) -> int:
        probs = self._probs()
        self._last_probs = probs
        return int(self._rng.choice(self.n_arms, p=probs))

    def update(self, action: int, reward: float) -> None:
        p_a = max(1e-12, float(self._last_probs[action]))
        reward_hat = float(reward) / p_a
        eta = self.gamma / float(self.n_arms)
        self.weights[action] *= np.exp(eta * reward_hat)


def _kl_bernoulli(p: float, q: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    q = float(np.clip(q, 1e-12, 1.0 - 1e-12))
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


class KLUCBAgent:
    """
    KL-UCB for Bernoulli rewards in ``[0, 1]``.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    c : float, default=3.0
        Exploration constant in confidence bound.
    seed : Optional[int], default=None
        RNG seed.
    """

    def __init__(self, n_arms: int, c: float = 3.0, seed: Optional[int] = None) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if c < 0.0:
            raise ValueError("c must be >= 0")
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.means = np.zeros(self.n_arms, dtype=np.float64)
        self.t = 0
        self._rng = np.random.default_rng(seed)

    def _bound(self, mean: float, n: int, t: int) -> float:
        if n <= 0:
            return 1.0
        limit = (np.log(max(2.0, float(t))) + self.c * np.log(max(2.0, np.log(max(2.0, float(t)))))) / float(n)
        lo, hi = float(mean), 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _kl_bernoulli(mean, mid) <= limit:
                lo = mid
            else:
                hi = mid
        return lo

    def select_action(self) -> int:
        self.t += 1
        untried = np.flatnonzero(self.counts == 0)
        if untried.size > 0:
            return int(self._rng.choice(untried))
        bounds = np.array([self._bound(self.means[a], int(self.counts[a]), self.t) for a in range(self.n_arms)])
        return int(self._rng.choice(np.flatnonzero(bounds == np.max(bounds))))

    def update(self, action: int, reward: float) -> None:
        reward = float(reward)
        if reward < 0.0 or reward > 1.0:
            raise ValueError("KLUCBAgent expects rewards in [0, 1]")
        self.counts[action] += 1
        n = float(self.counts[action])
        self.means[action] += (reward - self.means[action]) / n


class LinUCBAgent:
    """
    Linear UCB contextual bandit.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    context_dim : int
        Context feature dimension.
    alpha : float, default=1.0
        Confidence width multiplier.
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be > 0")
        if context_dim <= 0:
            raise ValueError("context_dim must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        self.n_arms = int(n_arms)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.A = np.stack([np.eye(self.context_dim, dtype=np.float64) for _ in range(self.n_arms)], axis=0)
        self.b = np.zeros((self.n_arms, self.context_dim), dtype=np.float64)

    def select_action(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError("context has wrong dimension")
        scores = np.zeros(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            exploit = float(theta_hat @ x)
            explore = self.alpha * float(np.sqrt(x @ A_inv @ x))
            scores[a] = exploit + explore
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError("context has wrong dimension")
        self.A[action] += np.outer(x, x)
        self.b[action] += float(reward) * x


_AGENT_REGISTRY = {
    "eps_greedy": EpsilonGreedyAgent,
    "optimistic_init": OptimisticInitialValuesAgent,
    "ucb": UCBAgent,
    "gradient_bandit": GradientBanditAgent,
    "thompson_sampling": ThompsonSamplingAgent,
    "exp3": EXP3Agent,
    "kl_ucb": KLUCBAgent,
    "linucb": LinUCBAgent,
}


def make_bandit_agent(name: str, n_arms: int, **kwargs: Any) -> BanditAgent:
    """
    Construct a bandit agent by registry name.

    Parameters
    ----------
    name : str
        Agent key, e.g. ``"eps_greedy"``, ``"ucb"``, ``"thompson_sampling"``.
    n_arms : int
        Number of actions.
    **kwargs : Any
        Additional constructor arguments for selected agent class.

    Returns
    -------
    BanditAgent
        Instantiated agent.
    """
    key = str(name).strip().lower()
    if key not in _AGENT_REGISTRY:
        available = ", ".join(sorted(_AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown bandit agent '{name}'. Available: {available}")
    return _AGENT_REGISTRY[key](n_arms=n_arms, **kwargs)


def run_bandit(
    env: Any,
    agent: BanditAgent,
    n_steps: int,
    seed: Optional[int] = None,
) -> BanditRunResult:
    """
    Run one bandit experiment for a fixed number of interaction steps.

    Parameters
    ----------
    env : Any
        Bandit-like environment exposing ``reset(seed=...)`` and
        ``step(action) -> (obs, reward, terminated, truncated, info)``.
    agent : BanditAgent
        Agent implementing ``select_action`` and ``update``.
    n_steps : int
        Number of interaction steps.
    seed : Optional[int], default=None
        Seed forwarded to ``env.reset``.

    Returns
    -------
    BanditRunResult
        Action and reward trajectories with aggregate metrics.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    obs, _ = env.reset(seed=seed)
    _ = obs

    actions = np.zeros(n_steps, dtype=np.int64)
    rewards = np.zeros(n_steps, dtype=np.float64)

    for t in range(n_steps):
        action = int(agent.select_action())
        _, reward, terminated, truncated, _ = env.step(action)
        agent.update(action, float(reward))

        actions[t] = action
        rewards[t] = float(reward)

        if terminated or truncated:
            env.reset()

    return BanditRunResult(
        actions=actions,
        rewards=rewards,
        mean_reward=float(np.mean(rewards)),
        cumulative_reward=float(np.sum(rewards)),
    )


def run_contextual_bandit(
    env: Any,
    agent: LinUCBAgent,
    n_steps: int,
    seed: Optional[int] = None,
) -> ContextualBanditRunResult:
    """
    Run one contextual-bandit experiment for a fixed number of steps.

    Parameters
    ----------
    env : Any
        Contextual-bandit env returning context as observation.
    agent : LinUCBAgent
        Contextual agent supporting ``select_action(context)`` and
        ``update(context, action, reward)``.
    n_steps : int
        Number of interaction steps.
    seed : Optional[int], default=None
        Seed forwarded to ``env.reset``.

    Returns
    -------
    ContextualBanditRunResult
        Action and reward trajectories with aggregate metrics.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    context, _ = env.reset(seed=seed)
    context = np.asarray(context, dtype=np.float64).reshape(-1)

    actions = np.zeros(n_steps, dtype=np.int64)
    rewards = np.zeros(n_steps, dtype=np.float64)

    for t in range(n_steps):
        action = int(agent.select_action(context))
        next_context, reward, terminated, truncated, _ = env.step(action)
        agent.update(context=context, action=action, reward=float(reward))
        actions[t] = action
        rewards[t] = float(reward)
        context = np.asarray(next_context, dtype=np.float64).reshape(-1)
        if terminated or truncated:
            context, _ = env.reset()
            context = np.asarray(context, dtype=np.float64).reshape(-1)

    return ContextualBanditRunResult(
        actions=actions,
        rewards=rewards,
        mean_reward=float(np.mean(rewards)),
        cumulative_reward=float(np.sum(rewards)),
    )
