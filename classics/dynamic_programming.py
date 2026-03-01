from __future__ import annotations
"""
Classic tabular dynamic-programming algorithms.

Includes policy evaluation, policy/value iteration, and modified policy
iteration over a fully known tabular MDP model.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class TabularMDP:
    """
    Finite tabular MDP model.

    Attributes
    ----------
    transition : np.ndarray
        Transition probabilities ``P[s, a, s']``, shape ``(S, A, S)``.
    reward : np.ndarray
        Expected immediate rewards ``R[s, a]``, shape ``(S, A)``.
    done : np.ndarray
        Terminal-transition mask ``done[s, a]``, shape ``(S, A)``.
    """

    transition: np.ndarray
    reward: np.ndarray
    done: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transition.shape[1])


@dataclass
class DPResult:
    """
    Output container for DP control algorithms.

    Attributes
    ----------
    values : np.ndarray
        Estimated optimal state values, shape ``(S,)``.
    policy : np.ndarray
        Greedy policy as discrete actions, shape ``(S,)``.
    q_values : np.ndarray
        Action-value table under final value function, shape ``(S, A)``.
    iterations : int
        Number of outer iterations performed.
    converged : bool
        Whether algorithm met convergence criterion.
    """

    values: np.ndarray
    policy: np.ndarray
    q_values: np.ndarray
    iterations: int
    converged: bool


def _state_from_row_col(width: int, row: int, col: int) -> int:
    return int(row) * int(width) + int(col)


def _to_tabular_mdp(
    transition: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
) -> TabularMDP:
    transition = np.asarray(transition, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done, dtype=np.bool_)

    if transition.ndim != 3:
        raise ValueError("transition must have shape (n_states, n_actions, n_states)")

    n_states, n_actions, n_states_2 = transition.shape
    if n_states != n_states_2:
        raise ValueError("transition last dimension must equal n_states")
    if reward.shape != (n_states, n_actions):
        raise ValueError("reward must have shape (n_states, n_actions)")
    if done.shape != (n_states, n_actions):
        raise ValueError("done must have shape (n_states, n_actions)")

    row_sums = transition.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("each transition[s, a, :] must sum to 1")

    return TabularMDP(transition=transition, reward=reward, done=done)


def build_tabular_mdp_from_env(
    env: Any,
    terminal_states: Optional[Sequence[int]] = None,
) -> TabularMDP:
    """
    Build a tabular MDP model from a deterministic env with Discrete spaces.

    The env is expected to expose ``observation_space.n``, ``action_space.n``,
    ``state_dict()``, and ``load_state_dict(...)``. This matches the tabular
    grid environments in ``classics/toy_env/tabular_grid.py``.

    Parameters
    ----------
    env : Any
        Environment with discrete state/action spaces and
        ``state_dict/load_state_dict`` hooks.
    terminal_states : Optional[Sequence[int]], default=None
        Optional explicit set of absorbing states.

    Returns
    -------
    TabularMDP
        Deterministic model converted into tabular tensor form.
    """
    if not hasattr(env, "observation_space") or not hasattr(env.observation_space, "n"):
        raise ValueError("env.observation_space.n is required")
    if not hasattr(env, "action_space") or not hasattr(env.action_space, "n"):
        raise ValueError("env.action_space.n is required")
    if not hasattr(env, "state_dict") or not hasattr(env, "load_state_dict"):
        raise ValueError("env must implement state_dict/load_state_dict for model extraction")

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    reward = np.zeros((n_states, n_actions), dtype=np.float64)
    done = np.zeros((n_states, n_actions), dtype=np.bool_)

    # Initialize env internals to obtain a template state payload.
    env.reset(seed=0)
    template_state = env.state_dict()

    if not all(k in template_state for k in ("row", "col", "step_count")):
        raise ValueError("env.state_dict() must include keys: row, col, step_count")

    width = int(getattr(env, "width", 0))
    if width <= 0:
        raise ValueError("env.width must be a positive integer")

    inferred_terminal_states = set()
    if terminal_states is not None:
        inferred_terminal_states = {int(s) for s in terminal_states}
    elif hasattr(env, "goal") and isinstance(getattr(env, "goal"), tuple):
        goal = getattr(env, "goal")
        if len(goal) == 2:
            inferred_terminal_states = {_state_from_row_col(width=width, row=goal[0], col=goal[1])}

    for s in range(n_states):
        row, col = divmod(s, width)
        for a in range(n_actions):
            if s in inferred_terminal_states:
                transition[s, a, s] = 1.0
                reward[s, a] = 0.0
                done[s, a] = True
                continue

            state = dict(template_state)
            state["row"] = int(row)
            state["col"] = int(col)
            state["step_count"] = 0
            env.load_state_dict(state)

            next_s, r, terminated, truncated, _ = env.step(a)
            if truncated:
                raise ValueError(
                    "Encountered truncated transition while building MDP. "
                    "Use a larger env.max_steps for planning."
                )

            next_s = int(next_s)
            transition[s, a, next_s] = 1.0
            reward[s, a] = float(r)
            done[s, a] = bool(terminated)

    return TabularMDP(transition=transition, reward=reward, done=done)


def _compute_q(values: np.ndarray, mdp: TabularMDP, gamma: float) -> np.ndarray:
    bootstrap = mdp.transition @ values
    return mdp.reward + gamma * bootstrap * (1.0 - mdp.done.astype(np.float64))


def policy_evaluation(
    policy: np.ndarray,
    transition: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    done: Optional[np.ndarray] = None,
    *,
    mdp: Optional[TabularMDP] = None,
    gamma: float = 0.99,
    tol: float = 1e-10,
    max_iters: int = 10_000,
) -> np.ndarray:
    """
    Evaluate a fixed tabular policy.

    Parameters
    ----------
    policy : np.ndarray
        Either deterministic actions ``(S,)`` or action probabilities ``(S, A)``.
    transition, reward, done : Optional[np.ndarray]
        Raw model arrays (ignored if ``mdp`` is provided).
    mdp : Optional[TabularMDP], keyword-only
        Pre-built MDP container.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    tol : float, default=1e-10
        Convergence threshold on max value change.
    max_iters : int, default=10000
        Maximum Bellman expectation updates.

    Returns
    -------
    np.ndarray
        State values ``V^\pi``, shape ``(S,)``.
    """
    if mdp is None:
        if transition is None or reward is None or done is None:
            raise ValueError("Provide either mdp or (transition, reward, done)")
        mdp = _to_tabular_mdp(transition=transition, reward=reward, done=done)
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")

    n_states, n_actions = mdp.n_states, mdp.n_actions
    p = np.asarray(policy)
    if p.shape == (n_states,):
        pi = np.zeros((n_states, n_actions), dtype=np.float64)
        pi[np.arange(n_states), p.astype(np.int64)] = 1.0
    elif p.shape == (n_states, n_actions):
        pi = p.astype(np.float64)
        row_sums = pi.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("policy rows must sum to positive values")
        pi /= row_sums
    else:
        raise ValueError("policy must have shape (n_states,) or (n_states, n_actions)")

    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iters):
        q_values = _compute_q(values, mdp, gamma)
        new_values = np.sum(pi * q_values, axis=1)
        if np.max(np.abs(new_values - values)) < tol:
            values = new_values
            break
        values = new_values
    return values


def policy_iteration(
    transition: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    done: Optional[np.ndarray] = None,
    *,
    mdp: Optional[TabularMDP] = None,
    gamma: float = 0.99,
    eval_tol: float = 1e-10,
    max_policy_eval_iters: int = 10_000,
    max_policy_improve_iters: int = 1_000,
) -> DPResult:
    """
    Solve a tabular MDP with policy iteration.

    Parameters
    ----------
    transition, reward, done : Optional[np.ndarray]
        Raw model arrays (ignored if ``mdp`` is provided).
    mdp : Optional[TabularMDP], keyword-only
        Pre-built MDP container.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    eval_tol : float, default=1e-10
        Evaluation convergence threshold.
    max_policy_eval_iters : int, default=10000
        Maximum sweeps for policy evaluation phase.
    max_policy_improve_iters : int, default=1000
        Maximum policy-improvement iterations.

    Returns
    -------
    DPResult
        Final value/policy pair and metadata.
    """
    if mdp is None:
        if transition is None or reward is None or done is None:
            raise ValueError("Provide either mdp or (transition, reward, done)")
        mdp = _to_tabular_mdp(transition=transition, reward=reward, done=done)

    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")

    n_states, n_actions = mdp.n_states, mdp.n_actions
    values = np.zeros(n_states, dtype=np.float64)
    policy = np.zeros(n_states, dtype=np.int64)

    converged = False
    improve_iter = 0

    for improve_iter in range(1, max_policy_improve_iters + 1):
        # Policy evaluation
        for _ in range(max_policy_eval_iters):
            old_values = values.copy()
            q_values = _compute_q(old_values, mdp, gamma)
            values = q_values[np.arange(n_states), policy]
            if np.max(np.abs(values - old_values)) < eval_tol:
                break

        # Policy improvement
        q_values = _compute_q(values, mdp, gamma)
        new_policy = np.argmax(q_values, axis=1).astype(np.int64)

        if np.array_equal(new_policy, policy):
            converged = True
            policy = new_policy
            break
        policy = new_policy

    q_values = _compute_q(values, mdp, gamma)
    return DPResult(
        values=values,
        policy=policy,
        q_values=q_values,
        iterations=improve_iter,
        converged=converged,
    )


def modified_policy_iteration(
    transition: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    done: Optional[np.ndarray] = None,
    *,
    mdp: Optional[TabularMDP] = None,
    gamma: float = 0.99,
    eval_iters: int = 5,
    tol: float = 1e-10,
    max_iters: int = 1_000,
) -> DPResult:
    """
    Solve a tabular MDP with modified policy iteration.

    Parameters
    ----------
    transition, reward, done : Optional[np.ndarray]
        Raw model arrays (ignored if ``mdp`` is provided).
    mdp : Optional[TabularMDP], keyword-only
        Pre-built MDP container.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    eval_iters : int, default=5
        Number of truncated policy-evaluation sweeps per outer iteration.
    tol : float, default=1e-10
        Bellman residual convergence threshold.
    max_iters : int, default=1000
        Maximum outer iterations.

    Returns
    -------
    DPResult
        Final value/policy pair and metadata.
    """
    if mdp is None:
        if transition is None or reward is None or done is None:
            raise ValueError("Provide either mdp or (transition, reward, done)")
        mdp = _to_tabular_mdp(transition=transition, reward=reward, done=done)
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")
    if eval_iters <= 0:
        raise ValueError("eval_iters must be > 0")

    n_states = mdp.n_states
    values = np.zeros(n_states, dtype=np.float64)
    policy = np.zeros(n_states, dtype=np.int64)
    converged = False

    it = 0
    for it in range(1, max_iters + 1):
        for _ in range(eval_iters):
            q_values = _compute_q(values, mdp, gamma)
            values = q_values[np.arange(n_states), policy]

        q_values = _compute_q(values, mdp, gamma)
        new_policy = np.argmax(q_values, axis=1).astype(np.int64)
        if np.array_equal(new_policy, policy):
            bellman_residual = np.max(np.abs(np.max(q_values, axis=1) - values))
            policy = new_policy
            if bellman_residual < tol:
                converged = True
                break
        policy = new_policy

    q_values = _compute_q(values, mdp, gamma)
    return DPResult(values=values, policy=policy, q_values=q_values, iterations=it, converged=converged)


def value_iteration(
    transition: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    done: Optional[np.ndarray] = None,
    *,
    mdp: Optional[TabularMDP] = None,
    gamma: float = 0.99,
    tol: float = 1e-10,
    max_iters: int = 10_000,
) -> DPResult:
    """
    Solve a tabular MDP with value iteration.

    Parameters
    ----------
    transition, reward, done : Optional[np.ndarray]
        Raw model arrays (ignored if ``mdp`` is provided).
    mdp : Optional[TabularMDP], keyword-only
        Pre-built MDP container.
    gamma : float, default=0.99
        Discount factor in ``[0, 1)``.
    tol : float, default=1e-10
        Convergence threshold on max value change.
    max_iters : int, default=10000
        Maximum value-iteration sweeps.

    Returns
    -------
    DPResult
        Final value/policy pair and metadata.
    """
    if mdp is None:
        if transition is None or reward is None or done is None:
            raise ValueError("Provide either mdp or (transition, reward, done)")
        mdp = _to_tabular_mdp(transition=transition, reward=reward, done=done)

    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")

    n_states = mdp.n_states
    values = np.zeros(n_states, dtype=np.float64)
    converged = False

    it = 0
    for it in range(1, max_iters + 1):
        q_values = _compute_q(values, mdp, gamma)
        new_values = np.max(q_values, axis=1)
        if np.max(np.abs(new_values - values)) < tol:
            values = new_values
            converged = True
            break
        values = new_values

    q_values = _compute_q(values, mdp, gamma)
    policy = np.argmax(q_values, axis=1).astype(np.int64)
    return DPResult(
        values=values,
        policy=policy,
        q_values=q_values,
        iterations=it,
        converged=converged,
    )
