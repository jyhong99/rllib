# `rllib/classics`

Reference implementations of classic reinforcement learning algorithms.

This package focuses on educational and baseline-friendly methods:

- tabular dynamic programming
- Monte Carlo prediction/control
- temporal-difference prediction/control
- policy-gradient (tabular softmax)
- model-based tabular planning/control
- multi-armed and contextual bandits
- Monte Carlo Tree Search (MCTS)

The implementations are NumPy-first, readable, and directly callable as plain Python functions.

---

## 1) Module Layout

```text
classics/
├─ __init__.py                # public API re-exports
├─ bandit.py                  # bandit + contextual bandit agents/runners
├─ dynamic_programming.py     # policy eval / policy iteration / value iteration
├─ monte_carlo.py             # first/every-visit MC + MC control variants
├─ temporal_difference.py     # TD(0), TD(lambda) prediction
├─ control_algorithms.py      # SARSA/Q-learning/Expected-SARSA + variants
├─ policy_gradient.py         # REINFORCE, actor-critic, natural PG, A2C
├─ model_based.py             # Dyna-Q, prioritized sweeping
├─ tree_search.py             # Monte Carlo Tree Search
└─ utils.py                   # shared tabular helpers
```

---

## 2) Public API at a Glance

Import from one place:

```python
import rllib.classics as rc
```

Main export groups:

- Bandits:
  - `EpsilonGreedyAgent`, `UCBAgent`, `KLUCBAgent`, `EXP3Agent`, `LinUCBAgent`, `GradientBanditAgent`, `ThompsonSamplingAgent`
  - `run_bandit`, `run_contextual_bandit`, `make_bandit_agent`
- Dynamic Programming:
  - `TabularMDP`, `build_tabular_mdp_from_env`
  - `policy_evaluation`, `policy_iteration`, `modified_policy_iteration`, `value_iteration`
- Monte Carlo:
  - `first_visit_mc_prediction`, `every_visit_mc_prediction`
  - `mc_control`, `mc_control_exploring_starts`, `off_policy_mc_control_importance_sampling`
- TD Prediction:
  - `td0_prediction`, `td_lambda_prediction`
- TD Control:
  - `sarsa`, `q_learning`, `double_q_learning`, `expected_sarsa`
  - `sarsa_lambda`, `expected_sarsa_lambda`, `watkins_q_lambda`
  - `n_step_sarsa`, `n_step_q_learning`
- Policy Gradient:
  - `reinforce`, `reinforce_with_baseline`, `actor_critic`, `natural_policy_gradient`, `a2c`
- Model-based:
  - `dyna_q`, `prioritized_sweeping`
- Tree Search:
  - `monte_carlo_tree_search`

---

## 3) Environment Assumptions

Most algorithms in this package expect **discrete tabular environments**:

- `env.observation_space.n`
- `env.action_space.n`
- Gym/Gymnasium-like `reset` / `step` interface returning `(obs, reward, terminated, truncated, info)` format

Additional requirements by module:

- Dynamic programming model extraction:
  - requires `env.state_dict()` and `env.load_state_dict(...)`
  - for some helpers, expects `env.width` and optionally `env.goal`
- MCTS:
  - requires `env.state_dict()` and `env.load_state_dict(...)` to simulate and restore states

If your environment does not expose these hooks, use direct model arrays (for DP) or wrap your env with adapter methods.

---

## 4) Return Types and Result Objects

Most functions return typed dataclasses so you can inspect full training traces.

Examples:

- `BanditRunResult`: actions, rewards, mean/cumulative reward
- `DPResult`: values, policy, q_values, convergence metadata
- `MCPredictionResult` / `MCControlResult`
- `TDPredictionResult` / `TDControlResult`
- `PolicyGradientResult`
- `ModelBasedResult`
- `MCTSResult`

Pattern:

```python
res = rc.q_learning(env, num_episodes=500)
print(res.q_values.shape, res.policy.shape, res.episode_returns.mean())
```

---

## 5) Quick Start Examples

### 5.1 TD Control (Q-learning)

```python
import gymnasium as gym
import rllib.classics as rc

env = gym.make("CliffWalking-v0")  # discrete state/action

res = rc.q_learning(
    env=env,
    num_episodes=2000,
    gamma=1.0,
    alpha=0.1,
    epsilon=0.1,
    seed=0,
)

print("Average return:", float(res.episode_returns.mean()))
print("Q table shape:", res.q_values.shape)
```

### 5.2 Dynamic Programming (Value Iteration from Tabular Model)

```python
import rllib.classics as rc

# if env supports state_dict/load_state_dict model extraction:
mdp = rc.build_tabular_mdp_from_env(env)

dp = rc.value_iteration(mdp=mdp, gamma=0.99, tol=1e-10)
print("Converged:", dp.converged, "Iterations:", dp.iterations)
print("Greedy policy shape:", dp.policy.shape)
```

### 5.3 Monte Carlo Prediction

```python
import numpy as np
import rllib.classics as rc

n_states = env.observation_space.n
policy = np.zeros(n_states, dtype=np.int64)  # deterministic action per state

pred = rc.first_visit_mc_prediction(
    env=env,
    policy=policy,
    num_episodes=5000,
    gamma=1.0,
    seed=1,
)

print("Value estimate shape:", pred.values.shape)
```

### 5.4 REINFORCE (Tabular Softmax Policy)

```python
import rllib.classics as rc

pg = rc.reinforce(
    env=env,
    num_episodes=2000,
    gamma=0.99,
    alpha=0.01,
    seed=7,
)

print("Policy matrix shape:", pg.policy.shape)
```

### 5.5 Bandit Run

```python
import rllib.classics as rc

agent = rc.UCBAgent(n_arms=10, c=2.0, seed=0)
res = rc.run_bandit(env=bandit_env, agent=agent, n_steps=5000)
print(res.mean_reward, res.cumulative_reward)
```

### 5.6 MCTS

```python
import rllib.classics as rc

mcts = rc.monte_carlo_tree_search(
    env=env,
    num_simulations=500,
    gamma=1.0,
    c_uct=1.4,
    seed=0,
)

print("Best action:", mcts.best_action)
print("Root policy:", mcts.policy)
```

---

## 6) Algorithm Families in Detail

### Bandits (`bandit.py`)

Included methods:

- epsilon-greedy
- optimistic initial values
- UCB1-style UCB
- gradient bandit
- Thompson sampling (Bernoulli-style posterior)
- EXP3
- KL-UCB
- LinUCB (contextual)

Utilities:

- `run_bandit(...)`: interaction loop for non-contextual agents
- `run_contextual_bandit(...)`: interaction loop for contextual agents
- `make_bandit_agent(...)`: name-based agent factory

### Dynamic Programming (`dynamic_programming.py`)

Model-based planning over explicit tabular MDP tensors:

- `policy_evaluation`
- `policy_iteration`
- `modified_policy_iteration`
- `value_iteration`

You can either:

- provide `transition`, `reward`, `done` arrays directly, or
- provide `mdp=TabularMDP(...)`.

### Monte Carlo (`monte_carlo.py`)

Prediction:

- first-visit MC
- every-visit MC

Control:

- epsilon-soft MC control
- exploring-starts MC control
- off-policy MC control with importance sampling

### Temporal-Difference Prediction (`temporal_difference.py`)

- `td0_prediction`
- `td_lambda_prediction` with accumulating traces

### Temporal-Difference Control (`control_algorithms.py`)

Core control:

- SARSA
- Q-learning
- Double Q-learning
- Expected SARSA

Extended variants:

- n-step SARSA / n-step Q-learning
- SARSA(lambda)
- Expected-SARSA(lambda)
- Watkins’ Q(lambda)

### Policy Gradient (`policy_gradient.py`)

Tabular softmax policy methods:

- REINFORCE
- REINFORCE with baseline
- Actor-Critic
- Natural Policy Gradient (tabular)
- A2C-style tabular update

### Model-Based (`model_based.py`)

- Dyna-Q
- Prioritized Sweeping

Both learn a tabular model from interaction and interleave planning updates.

### Tree Search (`tree_search.py`)

- Monte Carlo Tree Search (UCT-style selection + rollout evaluation)
- returns root-level policy, action values, visits, and tree size

---

## 7) Reproducibility

Most functions accept `seed`.

Recommended reproducibility practice:

1. Set algorithm seed
2. Use deterministic environment behavior where possible
3. Keep `max_steps` fixed across runs

---

## 8) Common Pitfalls

- **Using continuous environments**
  - most methods here require discrete state/action spaces
- **Policy shape mismatch**
  - prediction methods expect `(S,)` deterministic or `(S, A)` probabilities
- **Invalid gamma/alpha/epsilon**
  - functions validate ranges and will raise `ValueError`
- **DP model extraction failure**
  - ensure env implements `state_dict/load_state_dict` and required metadata
- **MCTS cannot restore state**
  - add state snapshot/restore support on env side

---

## 9) When to Use `classics` vs `model_free`

Use `classics` when:

- you want tabular/reference implementations
- you need lightweight experiments without deep network infrastructure
- you want clear algorithmic baselines for teaching or debugging

Use `model_free` when:

- you train neural policies/value functions
- you need replay, trainers, callbacks, logging stack, wrappers, distributed rollout

---

## 10) Extending `classics`

Suggested pattern for new algorithms:

1. Add a new module file in `classics/`
2. Define:
   - a result dataclass
   - core algorithm function(s)
   - strict input validation
3. Reuse helpers from `classics/utils.py` where possible
4. Re-export public symbols in `classics/__init__.py`
5. Add small deterministic smoke tests

---

## 11) Minimal Import Recipes

```python
import rllib.classics as rc
```

or selective:

```python
from rllib.classics import q_learning, value_iteration, reinforce, monte_carlo_tree_search
```

