from __future__ import annotations
"""Classic tabular policy-gradient and actor-critic methods."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .utils import default_horizon as _default_horizon
from .utils import validate_discrete_env as _validate_discrete_env


@dataclass
class PolicyGradientResult:
    """
    Output container for tabular policy-gradient methods.

    Attributes
    ----------
    theta : np.ndarray
        Policy parameter matrix (logits), shape ``(S, A)``.
    policy : np.ndarray
        Softmax policy probabilities, shape ``(S, A)``.
    episode_returns : np.ndarray
        Undiscounted episode returns over training.
    baseline_values : Optional[np.ndarray]
        Learned state-value baseline (if method uses critic/baseline).
    """

    theta: np.ndarray
    policy: np.ndarray
    episode_returns: np.ndarray
    baseline_values: Optional[np.ndarray] = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _sample_episode(
    env: Any,
    theta: np.ndarray,
    rng: np.random.Generator,
    max_steps: int,
) -> tuple[list[int], list[int], list[float], float]:
    state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    state = int(state)

    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    ep_return = 0.0

    for _ in range(max_steps):
        probs = _softmax(theta[state : state + 1])[0]
        action = int(rng.choice(theta.shape[1], p=probs))

        next_state, reward, terminated, truncated, _ = env.step(action)
        _ = next_state

        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
        ep_return += float(reward)

        state = int(next_state)
        if terminated or truncated:
            break

    return states, actions, rewards, ep_return


def reinforce(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.01,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_theta: Optional[np.ndarray] = None,
) -> PolicyGradientResult:
    """
    REINFORCE (Monte Carlo policy gradient) with tabular softmax policy.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.01
        Policy learning rate.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_theta : Optional[np.ndarray], default=None
        Optional initial policy logits.

    Returns
    -------
    PolicyGradientResult
        Learned policy parameters and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    if initial_theta is None:
        theta = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        theta = np.asarray(initial_theta, dtype=np.float64).copy()
        if theta.shape != (n_states, n_actions):
            raise ValueError("initial_theta must have shape (n_states, n_actions)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        states, actions, rewards, ep_return = _sample_episode(env, theta=theta, rng=rng, max_steps=int(max_steps))
        episode_returns[ep] = ep_return

        g = 0.0
        for t in range(len(states) - 1, -1, -1):
            s_t = states[t]
            a_t = actions[t]
            r_tp1 = rewards[t]
            g = float(gamma) * g + float(r_tp1)

            probs = _softmax(theta[s_t : s_t + 1])[0]
            grad_log = -probs
            grad_log[a_t] += 1.0
            theta[s_t] += float(alpha) * float(g) * grad_log

    policy = _softmax(theta)
    return PolicyGradientResult(
        theta=theta,
        policy=policy,
        episode_returns=episode_returns,
        baseline_values=None,
    )


def reinforce_with_baseline(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha_theta: float = 0.01,
    alpha_value: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_theta: Optional[np.ndarray] = None,
    initial_values: Optional[np.ndarray] = None,
) -> PolicyGradientResult:
    """
    REINFORCE with learned state-value baseline.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha_theta : float, default=0.01
        Policy learning rate.
    alpha_value : float, default=0.1
        Baseline value-function learning rate.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_theta : Optional[np.ndarray], default=None
        Optional initial policy logits.
    initial_values : Optional[np.ndarray], default=None
        Optional initial baseline values.

    Returns
    -------
    PolicyGradientResult
        Learned policy parameters, baseline values, and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha_theta <= 0.0:
        raise ValueError("alpha_theta must be > 0")
    if alpha_value <= 0.0:
        raise ValueError("alpha_value must be > 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    if initial_theta is None:
        theta = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        theta = np.asarray(initial_theta, dtype=np.float64).copy()
        if theta.shape != (n_states, n_actions):
            raise ValueError("initial_theta must have shape (n_states, n_actions)")

    if initial_values is None:
        baseline_values = np.zeros(n_states, dtype=np.float64)
    else:
        baseline_values = np.asarray(initial_values, dtype=np.float64).copy()
        if baseline_values.shape != (n_states,):
            raise ValueError("initial_values must have shape (n_states,)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        states, actions, rewards, ep_return = _sample_episode(env, theta=theta, rng=rng, max_steps=int(max_steps))
        episode_returns[ep] = ep_return

        g = 0.0
        for t in range(len(states) - 1, -1, -1):
            s_t = states[t]
            a_t = actions[t]
            r_tp1 = rewards[t]
            g = float(gamma) * g + float(r_tp1)

            advantage = float(g) - baseline_values[s_t]
            baseline_values[s_t] += float(alpha_value) * advantage

            probs = _softmax(theta[s_t : s_t + 1])[0]
            grad_log = -probs
            grad_log[a_t] += 1.0
            theta[s_t] += float(alpha_theta) * advantage * grad_log

    policy = _softmax(theta)
    return PolicyGradientResult(
        theta=theta,
        policy=policy,
        episode_returns=episode_returns,
        baseline_values=baseline_values,
    )


def actor_critic(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha_theta: float = 0.01,
    alpha_value: float = 0.1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_theta: Optional[np.ndarray] = None,
    initial_values: Optional[np.ndarray] = None,
) -> PolicyGradientResult:
    """
    One-step tabular Actor-Critic (softmax actor + state-value critic).

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha_theta : float, default=0.01
        Actor learning rate.
    alpha_value : float, default=0.1
        Critic learning rate.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_theta : Optional[np.ndarray], default=None
        Optional initial policy logits.
    initial_values : Optional[np.ndarray], default=None
        Optional initial critic values.

    Returns
    -------
    PolicyGradientResult
        Learned actor/critic parameters and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha_theta <= 0.0:
        raise ValueError("alpha_theta must be > 0")
    if alpha_value <= 0.0:
        raise ValueError("alpha_value must be > 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    if initial_theta is None:
        theta = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        theta = np.asarray(initial_theta, dtype=np.float64).copy()
        if theta.shape != (n_states, n_actions):
            raise ValueError("initial_theta must have shape (n_states, n_actions)")

    if initial_values is None:
        baseline_values = np.zeros(n_states, dtype=np.float64)
    else:
        baseline_values = np.asarray(initial_values, dtype=np.float64).copy()
        if baseline_values.shape != (n_states,):
            raise ValueError("initial_values must have shape (n_states,)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            probs = _softmax(theta[state : state + 1])[0]
            action = int(rng.choice(n_actions, p=probs))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            v_s = baseline_values[state]
            v_next = 0.0 if done else baseline_values[next_state]
            delta = float(reward) + float(gamma) * float(v_next) - float(v_s)

            baseline_values[state] += float(alpha_value) * delta

            grad_log = -probs
            grad_log[action] += 1.0
            theta[state] += float(alpha_theta) * delta * grad_log

            ep_return += float(reward)
            state = next_state
            if done:
                break

        episode_returns[ep] = ep_return

    policy = _softmax(theta)
    return PolicyGradientResult(
        theta=theta,
        policy=policy,
        episode_returns=episode_returns,
        baseline_values=baseline_values,
    )


def natural_policy_gradient(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.05,
    damping: float = 1e-3,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_theta: Optional[np.ndarray] = None,
) -> PolicyGradientResult:
    """
    Tabular natural policy gradient with state-local Fisher preconditioning.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of sampled episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha : float, default=0.05
        Natural-gradient step-size.
    damping : float, default=1e-3
        Fisher damping term for numerical stability.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_theta : Optional[np.ndarray], default=None
        Optional initial policy logits.

    Returns
    -------
    PolicyGradientResult
        Learned policy parameters and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if damping <= 0.0:
        raise ValueError("damping must be > 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)

    if initial_theta is None:
        theta = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        theta = np.asarray(initial_theta, dtype=np.float64).copy()
        if theta.shape != (n_states, n_actions):
            raise ValueError("initial_theta must have shape (n_states, n_actions)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        states, actions, rewards, ep_return = _sample_episode(env, theta=theta, rng=rng, max_steps=int(max_steps))
        episode_returns[ep] = ep_return

        g = 0.0
        for t in range(len(states) - 1, -1, -1):
            s_t = states[t]
            a_t = actions[t]
            g = float(gamma) * g + float(rewards[t])

            probs = _softmax(theta[s_t : s_t + 1])[0]
            grad_log = -probs
            grad_log[a_t] += 1.0

            fisher = np.diag(probs) - np.outer(probs, probs)
            natural_grad = np.linalg.solve(fisher + float(damping) * np.eye(n_actions), grad_log)
            theta[s_t] += float(alpha) * float(g) * natural_grad

    return PolicyGradientResult(theta=theta, policy=_softmax(theta), episode_returns=episode_returns)


def a2c(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha_theta: float = 0.01,
    alpha_value: float = 0.1,
    entropy_coef: float = 0.0,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    initial_theta: Optional[np.ndarray] = None,
    initial_values: Optional[np.ndarray] = None,
) -> PolicyGradientResult:
    """
    Tabular A2C-style actor-critic with entropy regularization.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    alpha_theta : float, default=0.01
        Actor learning rate.
    alpha_value : float, default=0.1
        Critic learning rate.
    entropy_coef : float, default=0.0
        Entropy regularization coefficient.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.
    initial_theta : Optional[np.ndarray], default=None
        Optional initial policy logits.
    initial_values : Optional[np.ndarray], default=None
        Optional initial critic values.

    Returns
    -------
    PolicyGradientResult
        Learned actor/critic parameters and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if alpha_theta <= 0.0 or alpha_value <= 0.0:
        raise ValueError("alpha_theta and alpha_value must be > 0")
    if entropy_coef < 0.0:
        raise ValueError("entropy_coef must be >= 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)

    if initial_theta is None:
        theta = np.zeros((n_states, n_actions), dtype=np.float64)
    else:
        theta = np.asarray(initial_theta, dtype=np.float64).copy()
        if theta.shape != (n_states, n_actions):
            raise ValueError("initial_theta must have shape (n_states, n_actions)")
    if initial_values is None:
        baseline_values = np.zeros(n_states, dtype=np.float64)
    else:
        baseline_values = np.asarray(initial_values, dtype=np.float64).copy()
        if baseline_values.shape != (n_states,):
            raise ValueError("initial_values must have shape (n_states,)")

    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        state = int(state)
        ep_return = 0.0
        for _ in range(int(max_steps)):
            probs = _softmax(theta[state : state + 1])[0]
            action = int(rng.choice(n_actions, p=probs))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = bool(terminated or truncated)

            v_next = 0.0 if done else baseline_values[next_state]
            delta = float(reward) + float(gamma) * float(v_next) - float(baseline_values[state])
            baseline_values[state] += float(alpha_value) * delta

            grad_log = -probs
            grad_log[action] += 1.0
            entropy_grad = -(-np.log(np.clip(probs, 1e-12, 1.0)) - 1.0)
            theta[state] += float(alpha_theta) * (delta * grad_log + float(entropy_coef) * entropy_grad)

            ep_return += float(reward)
            state = next_state
            if done:
                break

        episode_returns[ep] = ep_return

    return PolicyGradientResult(
        theta=theta,
        policy=_softmax(theta),
        episode_returns=episode_returns,
        baseline_values=baseline_values,
    )
