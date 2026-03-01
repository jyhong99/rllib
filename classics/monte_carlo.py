from __future__ import annotations
"""
Classic Monte Carlo prediction and control methods for discrete tabular tasks.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .utils import as_policy_probs as _as_policy_probs
from .utils import validate_discrete_env as _validate_discrete_env


@dataclass
class MCPredictionResult:
    """
    Output container for Monte Carlo prediction methods.

    Attributes
    ----------
    values : np.ndarray
        Estimated state values, shape ``(S,)``.
    returns_sum : np.ndarray
        Accumulated returns per state.
    returns_count : np.ndarray
        Number of return samples per state.
    """

    values: np.ndarray
    returns_sum: np.ndarray
    returns_count: np.ndarray


@dataclass
class MCControlResult:
    """
    Output container for Monte Carlo control methods.

    Attributes
    ----------
    q_values : np.ndarray
        Estimated action values, shape ``(S, A)``.
    policy : np.ndarray
        Final policy probabilities, shape ``(S, A)``.
    returns_sum : np.ndarray
        Accumulated returns/statistics per state-action.
    returns_count : np.ndarray
        Number of samples/statistics per state-action.
    """

    q_values: np.ndarray
    policy: np.ndarray
    returns_sum: np.ndarray
    returns_count: np.ndarray


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    n_states, n_actions = q_values.shape
    probs = np.full((n_states, n_actions), float(epsilon) / float(n_actions), dtype=np.float64)
    greedy = np.argmax(q_values + 1e-12 * rng.standard_normal(q_values.shape), axis=1)
    probs[np.arange(n_states), greedy] += 1.0 - float(epsilon)
    return probs


def _state_to_row_col(env: Any, state: int) -> Tuple[int, int]:
    width = int(getattr(env, "width", 0))
    if width <= 0:
        raise ValueError("env.width is required for exploring starts")
    row, col = divmod(int(state), width)
    return int(row), int(col)


def _sample_episode(
    env: Any,
    policy_probs: np.ndarray,
    rng: np.random.Generator,
    max_steps: Optional[int] = None,
    start_state: Optional[int] = None,
    start_action: Optional[int] = None,
) -> list[tuple[int, int, float]]:
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = int(getattr(env, "max_steps", n_states * 4))

    # Standard start
    state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    state = int(state)

    # Exploring starts override state if requested.
    if start_state is not None:
        if not hasattr(env, "state_dict") or not hasattr(env, "load_state_dict"):
            raise ValueError("env must implement state_dict/load_state_dict for exploring starts")
        payload = dict(env.state_dict())
        row, col = _state_to_row_col(env, start_state)
        payload["row"] = row
        payload["col"] = col
        payload["step_count"] = 0
        env.load_state_dict(payload)
        state = int(start_state)

    episode: list[tuple[int, int, float]] = []

    action = None
    if start_action is not None:
        action = int(start_action)

    for t in range(int(max_steps)):
        if action is None:
            action = int(rng.choice(n_actions, p=policy_probs[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, float(reward)))

        state = int(next_state)
        if terminated or truncated:
            break
        action = None

    return episode


def _mc_prediction(
    env: Any,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float,
    first_visit: bool,
    max_steps: Optional[int],
    seed: Optional[int],
) -> MCPredictionResult:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    policy_probs = _as_policy_probs(policy, n_states=n_states, n_actions=n_actions)
    returns_sum = np.zeros(n_states, dtype=np.float64)
    returns_count = np.zeros(n_states, dtype=np.float64)
    values = np.zeros(n_states, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for _ in range(num_episodes):
        episode = _sample_episode(env, policy_probs=policy_probs, rng=rng, max_steps=max_steps)
        g = 0.0
        visited: set[int] = set()

        for t in range(len(episode) - 1, -1, -1):
            s_t, _a_t, r_tp1 = episode[t]
            g = float(gamma) * g + float(r_tp1)
            if first_visit:
                if s_t in visited:
                    continue
                visited.add(s_t)

            returns_sum[s_t] += g
            returns_count[s_t] += 1.0
            values[s_t] = returns_sum[s_t] / returns_count[s_t]

    return MCPredictionResult(values=values, returns_sum=returns_sum, returns_count=returns_count)


def first_visit_mc_prediction(
    env: Any,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> MCPredictionResult:
    """
    First-visit Monte Carlo policy evaluation.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    policy : np.ndarray
        Deterministic ``(S,)`` or stochastic ``(S, A)`` policy.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.

    Returns
    -------
    MCPredictionResult
        Estimated values and return statistics.
    """
    return _mc_prediction(
        env=env,
        policy=policy,
        num_episodes=num_episodes,
        gamma=gamma,
        first_visit=True,
        max_steps=max_steps,
        seed=seed,
    )


def every_visit_mc_prediction(
    env: Any,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> MCPredictionResult:
    """
    Every-visit Monte Carlo policy evaluation.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    policy : np.ndarray
        Deterministic ``(S,)`` or stochastic ``(S, A)`` policy.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.

    Returns
    -------
    MCPredictionResult
        Estimated values and return statistics.
    """
    return _mc_prediction(
        env=env,
        policy=policy,
        num_episodes=num_episodes,
        gamma=gamma,
        first_visit=False,
        max_steps=max_steps,
        seed=seed,
    )


def mc_control(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    first_visit: bool = True,
) -> MCControlResult:
    """
    On-policy Monte Carlo control with epsilon-greedy improvement.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    epsilon : float, default=0.1
        Epsilon used for behavior/improvement policy.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    first_visit : bool, default=True
        Use first-visit or every-visit updates.

    Returns
    -------
    MCControlResult
        Learned Q-values and final epsilon-greedy policy.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    q_values = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_count = np.zeros((n_states, n_actions), dtype=np.float64)

    rng = np.random.default_rng(seed)

    for _ in range(num_episodes):
        policy_probs = _epsilon_greedy_probs(q_values, epsilon=epsilon, rng=rng)
        episode = _sample_episode(env, policy_probs=policy_probs, rng=rng, max_steps=max_steps)

        g = 0.0
        visited: set[tuple[int, int]] = set()

        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            g = float(gamma) * g + float(r_tp1)

            sa = (s_t, a_t)
            if first_visit:
                if sa in visited:
                    continue
                visited.add(sa)

            returns_sum[s_t, a_t] += g
            returns_count[s_t, a_t] += 1.0
            q_values[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]

    policy = _epsilon_greedy_probs(q_values, epsilon=epsilon, rng=rng)
    return MCControlResult(
        q_values=q_values,
        policy=policy,
        returns_sum=returns_sum,
        returns_count=returns_count,
    )


def mc_control_exploring_starts(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    first_visit: bool = True,
) -> MCControlResult:
    """
    Monte Carlo control with exploring starts.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API and ``state_dict/load_state_dict``.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    first_visit : bool, default=True
        Use first-visit or every-visit updates.

    Returns
    -------
    MCControlResult
        Learned Q-values and greedy target policy.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    q_values = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_count = np.zeros((n_states, n_actions), dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Start with any deterministic policy; ES ensures coverage via random starts.
    policy_probs = np.full((n_states, n_actions), 1.0 / float(n_actions), dtype=np.float64)

    for _ in range(num_episodes):
        start_state = int(rng.integers(0, n_states))
        start_action = int(rng.integers(0, n_actions))
        episode = _sample_episode(
            env,
            policy_probs=policy_probs,
            rng=rng,
            max_steps=max_steps,
            start_state=start_state,
            start_action=start_action,
        )

        g = 0.0
        visited: set[tuple[int, int]] = set()

        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            g = float(gamma) * g + float(r_tp1)

            sa = (s_t, a_t)
            if first_visit:
                if sa in visited:
                    continue
                visited.add(sa)

            returns_sum[s_t, a_t] += g
            returns_count[s_t, a_t] += 1.0
            q_values[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]

        # Greedy policy improvement.
        greedy_actions = np.argmax(q_values, axis=1)
        policy_probs = np.zeros((n_states, n_actions), dtype=np.float64)
        policy_probs[np.arange(n_states), greedy_actions] = 1.0

    return MCControlResult(
        q_values=q_values,
        policy=policy_probs,
        returns_sum=returns_sum,
        returns_count=returns_count,
    )


def off_policy_mc_control_importance_sampling(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    epsilon_behavior: float = 0.2,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    weighted: bool = True,
) -> MCControlResult:
    """
    Off-policy MC control via importance sampling.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    epsilon_behavior : float, default=0.2
        Epsilon for behavior policy (target policy is greedy).
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    weighted : bool, default=True
        If True, uses weighted importance sampling; otherwise ordinary IS.

    Returns
    -------
    MCControlResult
        Learned Q-values and greedy target policy.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 < epsilon_behavior <= 1.0):
        raise ValueError("epsilon_behavior must be in (0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    q_values = np.zeros((n_states, n_actions), dtype=np.float64)
    c_weights = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_count = np.zeros((n_states, n_actions), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for _ in range(num_episodes):
        behavior_probs = _epsilon_greedy_probs(q_values, epsilon=epsilon_behavior, rng=rng)
        episode = _sample_episode(env, policy_probs=behavior_probs, rng=rng, max_steps=max_steps)

        g = 0.0
        w = 1.0
        greedy_actions = np.argmax(q_values, axis=1)

        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            g = float(gamma) * g + float(r_tp1)

            if weighted:
                c_weights[s_t, a_t] += w
                q_values[s_t, a_t] += (w / max(c_weights[s_t, a_t], 1e-12)) * (g - q_values[s_t, a_t])
            else:
                returns_sum[s_t, a_t] += w * g
                returns_count[s_t, a_t] += w
                q_values[s_t, a_t] = returns_sum[s_t, a_t] / max(returns_count[s_t, a_t], 1e-12)

            greedy_actions[s_t] = int(np.argmax(q_values[s_t]))
            if a_t != greedy_actions[s_t]:
                break

            b_prob = float(behavior_probs[s_t, a_t])
            w = w / max(b_prob, 1e-12)

    target_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    target_policy[np.arange(n_states), np.argmax(q_values, axis=1)] = 1.0
    return MCControlResult(
        q_values=q_values,
        policy=target_policy,
        returns_sum=returns_sum if not weighted else c_weights,
        returns_count=returns_count if not weighted else np.ones_like(c_weights),
    )
