from __future__ import annotations
"""Tabular temporal-difference prediction methods."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .utils import as_policy_probs as _as_policy_probs
from .utils import default_horizon as _default_horizon
from .utils import validate_discrete_env as _validate_discrete_env


@dataclass
class TDPredictionResult:
    """
    Output container for TD prediction methods.

    Attributes
    ----------
    values : np.ndarray
        Estimated state values, shape ``(S,)``.
    episode_returns : np.ndarray
        Undiscounted episode returns over training, shape ``(N,)``.
    """

    values: np.ndarray
    episode_returns: np.ndarray


def td0_prediction(
    env: Any,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_values: Optional[np.ndarray] = None,
) -> TDPredictionResult:
    """
    Tabular TD(0) policy evaluation.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    policy : np.ndarray
        Deterministic ``(S,)`` or stochastic ``(S, A)`` policy.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Step-size for TD update.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_values : Optional[np.ndarray], default=None
        Optional initial value table.

    Returns
    -------
    TDPredictionResult
        Learned values and episode-return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    n_states, n_actions = _validate_discrete_env(env)
    policy_probs = _as_policy_probs(policy, n_states=n_states, n_actions=n_actions)

    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    if initial_values is None:
        values = np.zeros(n_states, dtype=np.float64)
    else:
        values = np.asarray(initial_values, dtype=np.float64).copy()
        if values.shape != (n_states,):
            raise ValueError("initial_values must have shape (n_states,)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = int(rng.choice(n_actions, p=policy_probs[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            bootstrap = 0.0 if (terminated or truncated) else values[next_state]
            td_target = float(reward) + float(gamma) * float(bootstrap)
            values[state] += float(alpha) * (td_target - values[state])

            ep_return += float(reward)
            state = next_state

            if terminated or truncated:
                break

        episode_returns[ep] = ep_return

    return TDPredictionResult(values=values, episode_returns=episode_returns)


def td_lambda_prediction(
    env: Any,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    lambda_: float = 0.9,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_values: Optional[np.ndarray] = None,
) -> TDPredictionResult:
    """
    Tabular TD(lambda) policy evaluation with accumulating traces.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    policy : np.ndarray
        Deterministic ``(S,)`` or stochastic ``(S, A)`` policy.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Step-size for TD update.
    lambda_ : float, default=0.9
        Eligibility trace decay in ``[0, 1]``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_values : Optional[np.ndarray], default=None
        Optional initial value table.

    Returns
    -------
    TDPredictionResult
        Learned values and episode-return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    policy_probs = _as_policy_probs(policy, n_states=n_states, n_actions=n_actions)

    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    if initial_values is None:
        values = np.zeros(n_states, dtype=np.float64)
    else:
        values = np.asarray(initial_values, dtype=np.float64).copy()
        if values.shape != (n_states,):
            raise ValueError("initial_values must have shape (n_states,)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        traces = np.zeros(n_states, dtype=np.float64)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = int(rng.choice(n_actions, p=policy_probs[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            bootstrap = 0.0 if (terminated or truncated) else values[next_state]
            delta = float(reward) + float(gamma) * float(bootstrap) - values[state]

            traces *= float(gamma) * float(lambda_)
            traces[state] += 1.0
            values += float(alpha) * float(delta) * traces

            ep_return += float(reward)
            state = next_state

            if terminated or truncated:
                break

        episode_returns[ep] = ep_return

    return TDPredictionResult(values=values, episode_returns=episode_returns)
