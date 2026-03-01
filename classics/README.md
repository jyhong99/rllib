# `rllib/classics`

Classic RL algorithms and toy environments for fast, tabular-first experimentation.

This package is intended for:

- learning/teaching core RL algorithms
- debugging algorithm logic with minimal moving parts
- building reliable baselines before scaling to deep RL stacks

---

## What is included

1. **Bandits**
2. **Dynamic Programming**
3. **Monte Carlo prediction/control**
4. **Temporal-Difference prediction/control**
5. **Policy Gradient (tabular softmax)**
6. **Model-based tabular control**
7. **Monte Carlo Tree Search**
8. **Toy environments** (`classics.toy_env`)

---

## Folder layout

```text
classics/
├─ __init__.py
├─ bandit.py
├─ dynamic_programming.py
├─ monte_carlo.py
├─ temporal_difference.py
├─ control_algorithms.py
├─ policy_gradient.py
├─ model_based.py
├─ tree_search.py
├─ utils.py
└─ toy_env/
   ├─ __init__.py
   ├─ multi_armed_bandit.py
   ├─ tabular_grid_base.py
   ├─ gridworld_env.py
   ├─ cliff_walking_env.py
   ├─ windy_gridworld_env.py
   └─ tabular_grid.py
```

---

## Public API overview

Main exports from `classics/__init__.py` include:

- Bandits:
  - `EpsilonGreedyAgent`, `OptimisticInitialValuesAgent`, `UCBAgent`, `KLUCBAgent`, `EXP3Agent`, `LinUCBAgent`, `GradientBanditAgent`, `ThompsonSamplingAgent`
  - `make_bandit_agent`, `run_bandit`, `run_contextual_bandit`
- DP:
  - `TabularMDP`, `DPResult`, `build_tabular_mdp_from_env`
  - `policy_evaluation`, `policy_iteration`, `modified_policy_iteration`, `value_iteration`
- MC:
  - `first_visit_mc_prediction`, `every_visit_mc_prediction`
  - `mc_control`, `mc_control_exploring_starts`, `off_policy_mc_control_importance_sampling`
- TD prediction:
  - `td0_prediction`, `td_lambda_prediction`
- TD control:
  - `sarsa`, `q_learning`, `double_q_learning`, `expected_sarsa`
  - `n_step_sarsa`, `n_step_q_learning`
  - `sarsa_lambda`, `expected_sarsa_lambda`, `watkins_q_lambda`
- Policy gradient:
  - `reinforce`, `reinforce_with_baseline`, `actor_critic`, `natural_policy_gradient`, `a2c`
- Model-based:
  - `dyna_q`, `prioritized_sweeping`
- Tree search:
  - `monte_carlo_tree_search`

---

## Toy environments (`classics.toy_env`)

`classics.toy_env` provides lightweight environments used by tabular algorithms:

- `GridworldEnv`
- `CliffWalkingEnv`
- `WindyGridworldEnv`
- `MultiArmedBanditEnv`

Registration helpers:

- `register_tabular_grid_envs()`
- `register_multi_armed_bandit_env()`

Registered entry points use:

- `classics.toy_env.tabular_grid:*`
- `classics.toy_env.multi_armed_bandit:*`

---

## Environment assumptions by algorithm family

Most algorithms in `classics` assume:

- `env.observation_space.n` exists
- `env.action_space.n` exists
- Gym/Gymnasium step API (`obs, reward, terminated, truncated, info`)

Additional requirements:

1. `dynamic_programming.build_tabular_mdp_from_env(...)`
   - requires `state_dict()` and `load_state_dict(...)`
   - expects env metadata like `width` (for tabular coordinate mapping)
2. `tree_search.monte_carlo_tree_search(...)`
   - requires `state_dict()` and `load_state_dict(...)` for simulation rollbacks

---

## Quickstart examples

### 1) Q-learning on tabular grid

```python
from classics.toy_env import GridworldEnv
from classics import q_learning

env = GridworldEnv(height=5, width=5)
res = q_learning(env, num_episodes=2000, gamma=1.0, alpha=0.1, epsilon=0.1, seed=0)

print(res.q_values.shape)        # (S, A)
print(res.episode_returns.mean())
```

### 2) Value iteration from extracted MDP model

```python
from classics.toy_env import GridworldEnv
from classics import build_tabular_mdp_from_env, value_iteration

env = GridworldEnv(height=5, width=5)
mdp = build_tabular_mdp_from_env(env)
dp = value_iteration(mdp=mdp, gamma=0.99)

print(dp.converged, dp.iterations)
print(dp.policy.shape)           # (S,)
```

### 3) Monte Carlo prediction

```python
import numpy as np
from classics.toy_env import CliffWalkingEnv
from classics import first_visit_mc_prediction

env = CliffWalkingEnv()
n_states = env.observation_space.n
random_policy = np.zeros((n_states, env.action_space.n), dtype=np.float64)
random_policy[:] = 1.0 / env.action_space.n

pred = first_visit_mc_prediction(env, random_policy, num_episodes=5000, gamma=1.0, seed=1)
print(pred.values.shape)
```

### 4) REINFORCE (tabular softmax)

```python
from classics.toy_env import WindyGridworldEnv
from classics import reinforce

env = WindyGridworldEnv()
pg = reinforce(env, num_episodes=3000, gamma=0.99, alpha=0.01, seed=7)
print(pg.policy.shape)           # (S, A)
```

### 5) Bandit run

```python
from classics.toy_env import MultiArmedBanditEnv
from classics import UCBAgent, run_bandit

env = MultiArmedBanditEnv()
agent = UCBAgent(n_arms=env.action_space.n, c=2.0, seed=0)
res = run_bandit(env=env, agent=agent, n_steps=5000)
print(res.mean_reward, res.cumulative_reward)
```

### 6) MCTS

```python
from classics.toy_env import GridworldEnv
from classics import monte_carlo_tree_search

env = GridworldEnv()
mcts = monte_carlo_tree_search(env, num_simulations=500, gamma=1.0, c_uct=1.4, seed=0)
print(mcts.best_action, mcts.policy)
```

---

## Result dataclasses

Algorithms return structured results with full traces, for example:

- `BanditRunResult`, `ContextualBanditRunResult`
- `DPResult`
- `MCPredictionResult`, `MCControlResult`
- `TDPredictionResult`, `TDControlResult`
- `PolicyGradientResult`
- `ModelBasedResult`
- `MCTSResult`

This makes post-analysis straightforward without digging into internal state.

---

## Choosing between `classics` and `model_free`

Use `classics` when:

- you want tabular or low-complexity baselines
- you need algorithmic clarity and small state spaces
- you are debugging update rules

Use `model_free` when:

- you need neural networks and modern deep RL training infrastructure
- you want replay/callback/logger/trainer abstractions at scale

---

## Extending `classics`

Recommended workflow:

1. Add a new module under `classics/`.
2. Define:
   - input validation
   - core algorithm function(s)
   - output dataclass
3. Reuse shared helpers from `classics/utils.py`.
4. Re-export public symbols in `classics/__init__.py`.
5. Add a deterministic smoke test script.

---

## Common pitfalls

1. Using non-discrete envs with tabular algorithms.
2. Passing policy arrays with wrong shape (`(S,)` vs `(S, A)`).
3. Forgetting `state_dict/load_state_dict` support for DP model extraction or MCTS.
4. Mismatched `max_steps` causing unintended truncation behavior.

