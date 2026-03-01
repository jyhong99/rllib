from __future__ import annotations
"""Monte Carlo Tree Search (MCTS) for discrete-action environments."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np


@dataclass
class MCTSResult:
    """
    Output container for Monte Carlo Tree Search at a root state.

    Attributes
    ----------
    best_action : int
        Root action with the highest visit count.
    action_visits : np.ndarray
        Visit counts per root action, shape ``(A,)``.
    action_values : np.ndarray
        Mean action value estimates per root action, shape ``(A,)``.
    policy : np.ndarray
        Visit-based policy over root actions, shape ``(A,)``.
    tree_size : int
        Number of expanded states in the search tree.
    """

    best_action: int
    action_visits: np.ndarray
    action_values: np.ndarray
    policy: np.ndarray
    tree_size: int


@dataclass
class _MCTSNode:
    visit_count: int
    action_visits: np.ndarray
    action_value_sums: np.ndarray


def _validate_discrete_action_env(env: Any) -> int:
    if not hasattr(env, "action_space") or not hasattr(env.action_space, "n"):
        raise ValueError("env.action_space.n is required")
    if not hasattr(env, "step"):
        raise ValueError("env.step is required")
    if not hasattr(env, "reset"):
        raise ValueError("env.reset is required")
    if not hasattr(env, "state_dict") or not hasattr(env, "load_state_dict"):
        raise ValueError("env.state_dict/load_state_dict are required for MCTS simulation")
    return int(env.action_space.n)


def _state_key(obs: Any, env: Any) -> Hashable:
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 0:
        obs_key: Hashable = obs_arr.item()
    else:
        obs_key = (obs_arr.shape, str(obs_arr.dtype), obs_arr.tobytes())

    state = env.state_dict()
    step_count = state.get("step_count", None) if isinstance(state, dict) else None
    return (obs_key, step_count)


def _select_ucb_action(node: _MCTSNode, c_uct: float, rng: np.random.Generator) -> int:
    unvisited = np.flatnonzero(node.action_visits == 0.0)
    if unvisited.size > 0:
        return int(rng.choice(unvisited))

    mean_values = node.action_value_sums / np.maximum(node.action_visits, 1.0)
    exploration = np.sqrt(np.log(float(node.visit_count) + 1.0) / node.action_visits)
    scores = mean_values + float(c_uct) * exploration
    best_actions = np.flatnonzero(scores == np.max(scores))
    return int(rng.choice(best_actions))


def _rollout(
    env: Any,
    gamma: float,
    max_rollout_steps: int,
    rng: np.random.Generator,
) -> float:
    g = 0.0
    discount = 1.0

    for _ in range(int(max_rollout_steps)):
        action = int(rng.integers(0, int(env.action_space.n)))
        _obs, reward, terminated, truncated, _info = env.step(action)
        g += discount * float(reward)
        if terminated or truncated:
            break
        discount *= float(gamma)

    return float(g)


def monte_carlo_tree_search(
    env: Any,
    num_simulations: int,
    gamma: float = 1.0,
    c_uct: float = 1.41421356237,
    max_depth: Optional[int] = None,
    rollout_max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> MCTSResult:
    """
    Run Monte Carlo Tree Search from the environment's current/reset state.

    Parameters
    ----------
    env : Any
        Environment with discrete ``action_space.n`` and Gym-like ``reset``/``step``.
        The environment must also implement ``state_dict`` and ``load_state_dict``.
    num_simulations : int
        Number of MCTS simulations to run.
    gamma : float, default=1.0
        Discount factor in ``[0, 1]``.
    c_uct : float, default=1.41421356237
        UCT exploration constant.
    max_depth : Optional[int], default=None
        Maximum tree depth per simulation. If ``None``, uses ``env.max_steps``
        when available, otherwise ``100``.
    rollout_max_steps : Optional[int], default=None
        Maximum number of random rollout steps after expansion. If ``None``,
        defaults to ``max_depth``.
    seed : Optional[int], default=None
        RNG seed used for tie-breaking and rollout sampling.

    Returns
    -------
    MCTSResult
        Root action recommendation and visit/value statistics.
    """
    if num_simulations <= 0:
        raise ValueError("num_simulations must be > 0")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if c_uct < 0.0:
        raise ValueError("c_uct must be >= 0")

    n_actions = _validate_discrete_action_env(env)
    if max_depth is None:
        max_depth = int(getattr(env, "max_steps", 100))
    if max_depth <= 0:
        raise ValueError("max_depth must be > 0")
    if rollout_max_steps is None:
        rollout_max_steps = int(max_depth)
    if rollout_max_steps < 0:
        raise ValueError("rollout_max_steps must be >= 0")

    rng = np.random.default_rng(seed)
    root_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    root_snapshot = dict(env.state_dict())
    root_key = _state_key(root_obs, env)

    tree: Dict[Hashable, _MCTSNode] = {
        root_key: _MCTSNode(
            visit_count=0,
            action_visits=np.zeros(n_actions, dtype=np.float64),
            action_value_sums=np.zeros(n_actions, dtype=np.float64),
        )
    }

    for _ in range(num_simulations):
        env.load_state_dict(dict(root_snapshot))
        obs = root_obs
        path: list[Tuple[Hashable, int, float]] = []
        leaf_value = 0.0

        for depth in range(int(max_depth)):
            key = _state_key(obs, env)
            node = tree[key]
            action = _select_ucb_action(node=node, c_uct=c_uct, rng=rng)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            path.append((key, action, float(reward)))
            done = bool(terminated or truncated)
            if done:
                leaf_value = 0.0
                break

            next_key = _state_key(next_obs, env)
            if next_key not in tree:
                tree[next_key] = _MCTSNode(
                    visit_count=0,
                    action_visits=np.zeros(n_actions, dtype=np.float64),
                    action_value_sums=np.zeros(n_actions, dtype=np.float64),
                )
                remaining = int(max(0, rollout_max_steps - depth - 1))
                leaf_value = _rollout(env=env, gamma=gamma, max_rollout_steps=remaining, rng=rng)
                break

            obs = next_obs
        else:
            leaf_value = 0.0

        g = float(leaf_value)
        for key, action, reward in reversed(path):
            g = float(reward) + float(gamma) * g
            node = tree[key]
            node.visit_count += 1
            node.action_visits[action] += 1.0
            node.action_value_sums[action] += g

    root = tree[root_key]
    visits = root.action_visits.copy()
    total_visits = float(np.sum(visits))
    if total_visits <= 0.0:
        policy = np.full(n_actions, 1.0 / float(n_actions), dtype=np.float64)
    else:
        policy = visits / total_visits
    action_values = np.divide(
        root.action_value_sums,
        np.maximum(root.action_visits, 1.0),
    )
    best_action = int(np.argmax(visits if total_visits > 0.0 else action_values))

    return MCTSResult(
        best_action=best_action,
        action_visits=visits,
        action_values=action_values,
        policy=policy,
        tree_size=len(tree),
    )
