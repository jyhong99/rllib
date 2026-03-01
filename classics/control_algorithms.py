from __future__ import annotations
"""Tabular temporal-difference control algorithms."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .utils import default_horizon as _default_horizon
from .utils import epsilon_greedy_action as _epsilon_greedy_action
from .utils import greedy_policy_from_q as _greedy_policy_from_q
from .utils import validate_discrete_env as _validate_discrete_env


@dataclass
class TDControlResult:
    """
    Output container for tabular TD control methods.

    Attributes
    ----------
    q_values : np.ndarray
        Learned action-value table, shape ``(S, A)``.
    policy : np.ndarray
        Greedy policy probabilities, shape ``(S, A)``.
    episode_returns : np.ndarray
        Undiscounted episode returns over training.
    """

    q_values: np.ndarray
    policy: np.ndarray
    episode_returns: np.ndarray


def _epsilon_greedy_probs_for_state(
    q_values: np.ndarray,
    state: int,
    epsilon: float,
) -> np.ndarray:
    n_actions = int(q_values.shape[1])
    probs = np.full(n_actions, float(epsilon) / float(n_actions), dtype=np.float64)
    greedy_action = int(np.argmax(q_values[state]))
    probs[greedy_action] += 1.0 - float(epsilon)
    return probs


def _validate_inputs(num_episodes: int, gamma: float, alpha: float, epsilon: float) -> None:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")


def _validate_n_step(n_step: int) -> int:
    n = int(n_step)
    if n <= 0:
        raise ValueError("n_step must be > 0")
    return n


def _init_q(initial_q: Optional[np.ndarray], n_states: int, n_actions: int) -> np.ndarray:
    if initial_q is None:
        return np.zeros((n_states, n_actions), dtype=np.float64)
    q_values = np.asarray(initial_q, dtype=np.float64).copy()
    if q_values.shape != (n_states, n_actions):
        raise ValueError("initial_q must have shape (n_states, n_actions)")
    return q_values


def sarsa(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular on-policy SARSA control.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            if done:
                td_target = float(reward)
                q_values[state, action] += float(alpha) * (td_target - q_values[state, action])
                ep_return += float(reward)
                break

            next_action = _epsilon_greedy_action(q_values, next_state, epsilon=epsilon, rng=rng)
            td_target = float(reward) + float(gamma) * q_values[next_state, next_action]
            q_values[state, action] += float(alpha) * (td_target - q_values[state, action])

            ep_return += float(reward)
            state, action = next_state, next_action

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def q_learning(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular off-policy Q-learning.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            if done:
                td_target = float(reward)
            else:
                td_target = float(reward) + float(gamma) * float(np.max(q_values[next_state]))

            q_values[state, action] += float(alpha) * (td_target - q_values[state, action])
            ep_return += float(reward)
            state = next_state

            if done:
                break

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def expected_sarsa(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular Expected SARSA control.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior/target policy.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            if done:
                expected_next = 0.0
            else:
                probs = _epsilon_greedy_probs_for_state(q_values, next_state, epsilon=epsilon)
                expected_next = float(np.dot(probs, q_values[next_state]))

            td_target = float(reward) + float(gamma) * float(expected_next)
            q_values[state, action] += float(alpha) * (td_target - q_values[state, action])

            ep_return += float(reward)
            state = next_state
            if done:
                break

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def n_step_sarsa(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    n_step: int = 3,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular n-step SARSA.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    n_step : int, default=3
        Backup horizon ``n``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_step = _validate_n_step(n_step)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        s0, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        s0 = int(s0)
        a0 = _epsilon_greedy_action(q_values, s0, epsilon=epsilon, rng=rng)

        states = [s0]
        actions = [a0]
        rewards = [0.0]  # rewards[t] corresponds to R_t

        t = 0
        T = np.inf
        ep_return = 0.0

        while True:
            if t < T:
                s_t = states[t]
                a_t = actions[t]
                s_next, r_next, terminated, truncated, _ = env.step(a_t)
                s_next = int(s_next)
                rewards.append(float(r_next))
                states.append(s_next)
                ep_return += float(r_next)

                done = bool(terminated or truncated)
                if done:
                    T = t + 1
                else:
                    a_next = _epsilon_greedy_action(q_values, s_next, epsilon=epsilon, rng=rng)
                    actions.append(a_next)

            tau = t - n_step + 1
            if tau >= 0:
                end = tau + n_step if np.isinf(T) else min(tau + n_step, int(T))
                g = 0.0
                for i in range(tau + 1, end + 1):
                    g += (float(gamma) ** (i - tau - 1)) * rewards[i]

                if tau + n_step < T:
                    s_boot = states[tau + n_step]
                    a_boot = actions[tau + n_step]
                    g += (float(gamma) ** n_step) * q_values[s_boot, a_boot]

                s_tau = states[tau]
                a_tau = actions[tau]
                q_values[s_tau, a_tau] += float(alpha) * (g - q_values[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def n_step_q_learning(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    n_step: int = 3,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular n-step Q-learning.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    n_step : int, default=3
        Backup horizon ``n``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_step = _validate_n_step(n_step)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        s0, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        s0 = int(s0)

        states = [s0]
        actions: list[int] = []
        rewards = [0.0]  # rewards[t] corresponds to R_t

        t = 0
        T = np.inf
        ep_return = 0.0

        while True:
            if t < T:
                s_t = states[t]
                a_t = _epsilon_greedy_action(q_values, s_t, epsilon=epsilon, rng=rng)
                actions.append(a_t)
                s_next, r_next, terminated, truncated, _ = env.step(a_t)
                s_next = int(s_next)
                rewards.append(float(r_next))
                states.append(s_next)
                ep_return += float(r_next)

                if terminated or truncated:
                    T = t + 1

            tau = t - n_step + 1
            if tau >= 0:
                end = tau + n_step if np.isinf(T) else min(tau + n_step, int(T))
                g = 0.0
                for i in range(tau + 1, end + 1):
                    g += (float(gamma) ** (i - tau - 1)) * rewards[i]

                if tau + n_step < T:
                    s_boot = states[tau + n_step]
                    g += (float(gamma) ** n_step) * float(np.max(q_values[s_boot]))

                s_tau = states[tau]
                a_tau = actions[tau]
                q_values[s_tau, a_tau] += float(alpha) * (g - q_values[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def sarsa_lambda(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lambda_: float = 0.9,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular SARSA(lambda) with accumulating traces.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    lambda_ : float, default=0.9
        Trace-decay parameter.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        traces = np.zeros((n_states, n_actions), dtype=np.float64)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            if done:
                next_q = 0.0
            else:
                next_action = _epsilon_greedy_action(q_values, next_state, epsilon=epsilon, rng=rng)
                next_q = float(q_values[next_state, next_action])

            delta = float(reward) + float(gamma) * next_q - float(q_values[state, action])
            traces *= float(gamma) * float(lambda_)
            traces[state, action] += 1.0
            q_values += float(alpha) * delta * traces

            ep_return += float(reward)
            if done:
                break

            state, action = next_state, next_action

        episode_returns[ep] = ep_return

    return TDControlResult(
        q_values=q_values,
        policy=_greedy_policy_from_q(q_values),
        episode_returns=episode_returns,
    )


def expected_sarsa_lambda(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lambda_: float = 0.9,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular Expected SARSA(lambda) with accumulating traces.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior/target policy.
    lambda_ : float, default=0.9
        Trace-decay parameter.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        traces = np.zeros((n_states, n_actions), dtype=np.float64)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)
            if done:
                expected_next = 0.0
            else:
                probs = _epsilon_greedy_probs_for_state(q_values, next_state, epsilon=epsilon)
                expected_next = float(np.dot(probs, q_values[next_state]))

            delta = float(reward) + float(gamma) * expected_next - float(q_values[state, action])
            traces *= float(gamma) * float(lambda_)
            traces[state, action] += 1.0
            q_values += float(alpha) * delta * traces

            ep_return += float(reward)
            state = next_state
            if done:
                break

        episode_returns[ep] = ep_return

    return TDControlResult(q_values=q_values, policy=_greedy_policy_from_q(q_values), episode_returns=episode_returns)


def watkins_q_lambda(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lambda_: float = 0.9,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_q: Optional[np.ndarray] = None,
) -> TDControlResult:
    """
    Tabular Watkins Q(lambda).

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    lambda_ : float, default=0.9
        Trace-decay parameter.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_q : Optional[np.ndarray], default=None
        Optional initial Q table.

    Returns
    -------
    TDControlResult
        Learned Q-values, greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0, 1]")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    q_values = _init_q(initial_q, n_states=n_states, n_actions=n_actions)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        traces = np.zeros((n_states, n_actions), dtype=np.float64)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            action = _epsilon_greedy_action(q_values, state, epsilon=epsilon, rng=rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            greedy_next_action = int(np.argmax(q_values[next_state])) if not done else 0
            max_next = 0.0 if done else float(q_values[next_state, greedy_next_action])
            delta = float(reward) + float(gamma) * max_next - float(q_values[state, action])

            traces *= float(gamma) * float(lambda_)
            traces[state, action] += 1.0
            q_values += float(alpha) * delta * traces

            # Watkins reset when behavior action is not greedy.
            if not done:
                if action != int(np.argmax(q_values[state])):
                    traces.fill(0.0)

            ep_return += float(reward)
            state = next_state
            if done:
                break

        episode_returns[ep] = ep_return

    return TDControlResult(q_values=q_values, policy=_greedy_policy_from_q(q_values), episode_returns=episode_returns)


def double_q_learning(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> TDControlResult:
    """
    Tabular Double Q-learning.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy on ``Q1 + Q2``.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.

    Returns
    -------
    TDControlResult
        Learned Q-values (``Q1+Q2``), greedy policy, and return trace.
    """
    _validate_inputs(num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)

    q1 = np.zeros((n_states, n_actions), dtype=np.float64)
    q2 = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0
        for _ in range(int(max_steps)):
            q_sum = q1 + q2
            action = _epsilon_greedy_action(q_sum, state, epsilon=epsilon, rng=rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            if rng.random() < 0.5:
                if done:
                    target = float(reward)
                else:
                    a_star = int(np.argmax(q1[next_state]))
                    target = float(reward) + float(gamma) * float(q2[next_state, a_star])
                q1[state, action] += float(alpha) * (target - q1[state, action])
            else:
                if done:
                    target = float(reward)
                else:
                    a_star = int(np.argmax(q2[next_state]))
                    target = float(reward) + float(gamma) * float(q1[next_state, a_star])
                q2[state, action] += float(alpha) * (target - q2[state, action])

            ep_return += float(reward)
            state = next_state
            if done:
                break
        episode_returns[ep] = ep_return

    q_values = q1 + q2
    return TDControlResult(q_values=q_values, policy=_greedy_policy_from_q(q_values), episode_returns=episode_returns)
