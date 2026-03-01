from __future__ import annotations
"""Classic tabular model-based control methods."""

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from .utils import default_horizon as _default_horizon
from .utils import epsilon_greedy_action as _epsilon_greedy_action
from .utils import greedy_policy_from_q as _greedy_policy_from_q
from .utils import validate_discrete_env as _validate_discrete_env


@dataclass
class ModelBasedResult:
    """
    Output container for tabular model-based control methods.

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


def dyna_q(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    planning_steps: int = 10,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> ModelBasedResult:
    """
    Tabular Dyna-Q.

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    planning_steps : int, default=10
        Number of model rollouts per real step.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.

    Returns
    -------
    ModelBasedResult
        Learned Q-values/policy and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if planning_steps < 0:
        raise ValueError("planning_steps must be >= 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        s = int(s)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            a = _epsilon_greedy_action(q, s, epsilon=epsilon, rng=rng)
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = int(s2)
            done = bool(terminated or truncated)

            target = float(r) if done else float(r) + float(gamma) * float(np.max(q[s2]))
            q[s, a] += float(alpha) * (target - q[s, a])
            model[(s, a)] = (float(r), s2, done)
            ep_return += float(r)

            if model and planning_steps > 0:
                keys = list(model.keys())
                for _k in range(planning_steps):
                    s_m, a_m = keys[int(rng.integers(0, len(keys)))]
                    r_m, s2_m, done_m = model[(s_m, a_m)]
                    target_m = r_m if done_m else r_m + float(gamma) * float(np.max(q[s2_m]))
                    q[s_m, a_m] += float(alpha) * (target_m - q[s_m, a_m])

            s = s2
            if done:
                break
        episode_returns[ep] = ep_return

    return ModelBasedResult(q_values=q, policy=_greedy_policy_from_q(q), episode_returns=episode_returns)


def prioritized_sweeping(
    env: Any,
    num_episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    planning_steps: int = 10,
    theta: float = 1e-5,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> ModelBasedResult:
    """
    Tabular prioritized sweeping (deterministic model).

    Parameters
    ----------
    env : Any
        Discrete env with Gym-like API.
    num_episodes : int
        Number of training episodes.
    gamma : float, default=1.0
        Discount factor.
    alpha : float, default=0.1
        Learning rate.
    epsilon : float, default=0.1
        Epsilon for behavior policy.
    planning_steps : int, default=10
        Number of queue updates per real step.
    theta : float, default=1e-5
        Priority threshold.
    max_steps : Optional[int], default=None
        Episode horizon override.
    seed : Optional[int], default=None
        RNG seed.

    Returns
    -------
    ModelBasedResult
        Learned Q-values/policy and return trace.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if planning_steps < 0:
        raise ValueError("planning_steps must be >= 0")
    if theta < 0.0:
        raise ValueError("theta must be >= 0")

    n_states, n_actions = _validate_discrete_env(env)
    if max_steps is None:
        max_steps = _default_horizon(env, n_states)

    q = np.zeros((n_states, n_actions), dtype=np.float64)
    model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    predecessors: Dict[int, Set[Tuple[int, int]]] = {}
    pq: list[Tuple[float, Tuple[int, int]]] = []
    episode_returns = np.zeros(num_episodes, dtype=np.float64)
    rng = np.random.default_rng(seed)

    def _priority(s: int, a: int, r: float, s2: int, done: bool) -> float:
        target = r if done else r + float(gamma) * float(np.max(q[s2]))
        return abs(target - float(q[s, a]))

    for ep in range(num_episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        s = int(s)
        ep_return = 0.0

        for _ in range(int(max_steps)):
            a = _epsilon_greedy_action(q, s, epsilon=epsilon, rng=rng)
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = int(s2)
            done = bool(terminated or truncated)

            p = _priority(s, a, float(r), s2, done)
            if p > theta:
                heappush(pq, (-p, (s, a)))

            model[(s, a)] = (float(r), s2, done)
            predecessors.setdefault(s2, set()).add((s, a))

            for _plan in range(planning_steps):
                if not pq:
                    break
                _neg_p, (sp, ap) = heappop(pq)
                rp, s2p, donep = model[(sp, ap)]
                targetp = rp if donep else rp + float(gamma) * float(np.max(q[s2p]))
                q[sp, ap] += float(alpha) * (targetp - q[sp, ap])

                for s_pre, a_pre in predecessors.get(sp, set()):
                    r_pre, s2_pre, done_pre = model[(s_pre, a_pre)]
                    p_pre = _priority(s_pre, a_pre, r_pre, s2_pre, done_pre)
                    if p_pre > theta:
                        heappush(pq, (-p_pre, (s_pre, a_pre)))

            ep_return += float(r)
            s = s2
            if done:
                break
        episode_returns[ep] = ep_return

    return ModelBasedResult(q_values=q, policy=_greedy_policy_from_q(q), episode_returns=episode_returns)
