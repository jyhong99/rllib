from __future__ import annotations
"""Shared utility helpers for classic tabular algorithms."""

from typing import Any

import numpy as np


def validate_discrete_env(env: Any) -> tuple[int, int]:
    if not hasattr(env, "observation_space") or not hasattr(env.observation_space, "n"):
        raise ValueError("env.observation_space.n is required")
    if not hasattr(env, "action_space") or not hasattr(env.action_space, "n"):
        raise ValueError("env.action_space.n is required")
    return int(env.observation_space.n), int(env.action_space.n)


def default_horizon(env: Any, n_states: int) -> int:
    if hasattr(env, "max_steps"):
        return int(getattr(env, "max_steps"))
    return int(n_states * 4)


def as_policy_probs(policy: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
    p = np.asarray(policy)
    if p.shape == (n_states,):
        probs = np.zeros((n_states, n_actions), dtype=np.float64)
        probs[np.arange(n_states), p.astype(np.int64)] = 1.0
        return probs

    if p.shape == (n_states, n_actions):
        probs = p.astype(np.float64)
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0.0):
            raise ValueError("policy rows must sum to a positive value")
        return probs / row_sums[:, None]

    raise ValueError("policy must have shape (n_states,) or (n_states, n_actions)")


def epsilon_greedy_action(
    q_values: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    n_actions = int(q_values.shape[1])
    if rng.random() < float(epsilon):
        return int(rng.integers(0, n_actions))

    row = q_values[state]
    max_q = float(np.max(row))
    candidates = np.flatnonzero(row == max_q)
    return int(rng.choice(candidates))


def greedy_policy_from_q(q_values: np.ndarray) -> np.ndarray:
    n_states, n_actions = q_values.shape
    policy = np.zeros((n_states, n_actions), dtype=np.float64)
    policy[np.arange(n_states), np.argmax(q_values, axis=1)] = 1.0
    return policy
