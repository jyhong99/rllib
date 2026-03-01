# `rllib/model_free`

Comprehensive model-free RL library with a shared, production-oriented training stack.

This module contains:

- modern **value-based** algorithms (DQN family, distributional variants, recurrent variants)
- **policy-based** on-policy and off-policy actor(-critic) algorithms
- **offline RL** baselines
- a common infrastructure layer for buffers, trainers, callbacks, logging, wrappers, and optimization utilities

The core design is compositional:

- `Head`: network/inference object
- `Core`: update/loss engine
- `PolicyAlgorithm`: rollout/replay driver
- `Trainer`: end-to-end orchestration

---

## 1) Feature Summary

- Unified training interfaces via `Trainer` / `build_trainer(...)`
- Single-environment and Ray multi-worker execution paths
- Replay variants: standard replay, PER, HER, sequence replay
- Logging stack: TensorBoard / CSV / JSONL / W&B / stdout
- Transition-level JSONL logging (toggle at trainer level)
- Callback system for eval, checkpointing, early stop, NaN-guard, timing, LR logging, and more
- Optional wrappers for Atari preprocessing and observation/reward normalization

---

## 2) Directory Layout

```text
model_free/
├─ baselines/
│  ├─ value_based/
│  │  ├─ dqn/
│  │  ├─ drqn/
│  │  ├─ c51/
│  │  ├─ qrdqn/
│  │  ├─ iqn/
│  │  ├─ fqf/
│  │  └─ rainbow/
│  ├─ policy_based/
│  │  ├─ on_policy/
│  │  │  ├─ a2c/ a2c_discrete/ acktr/ ppo/ ppo_discrete/ trpo/ vpg/ vpg_discrete/
│  │  └─ off_policy/
│  │     ├─ ddpg/ td3/ sac/ sac_discrete/ tqc/ redq/ acer/
│  └─ offline/
│     ├─ cql/
│     ├─ iql/
│     └─ td3bc/
└─ common/
   ├─ buffers/         # replay/rollout/PER/HER
   ├─ callbacks/       # trainer hook stack
   ├─ loggers/         # scalar logger + writer backends + transition logger
   ├─ networks/        # shared network primitives
   ├─ noises/          # action/exploration noise
   ├─ optimizers/      # optimizer/scheduler builders (+ custom opts)
   ├─ policies/        # on/off-policy drivers + core base classes
   ├─ regularizations/ # augmentation/regularization utilities
   ├─ trainers/        # trainer orchestrator + single/ray loops
   ├─ utils/           # shared utility modules
   └─ wrappers/        # normalize and atari wrappers
```

---

## 3) Algorithm Catalog

### Value-based
- `dqn`
- `drqn`
- `c51`
- `qrdqn`
- `iqn`
- `fqf`
- `rainbow`

### Policy-based Off-policy
- `ddpg`
- `td3`
- `sac`
- `sac_discrete`
- `tqc`
- `redq`
- `acer`

### Policy-based On-policy
- `a2c`
- `a2c_discrete`
- `acktr`
- `ppo`
- `ppo_discrete`
- `trpo`
- `vpg`
- `vpg_discrete`

### Offline
- `cql`
- `iql`
- `td3bc`

---

## 4) Installation Notes

This repository does not enforce one single environment definition in this folder, but typical runtime dependencies are:

- Python 3.9+
- `torch`
- `numpy`
- `gymnasium`
- `tqdm`
- optional: `ray`, `tensorboard`, `wandb`

If you use Ray multi-worker training, install Ray explicitly.

---

## 5) Quick Start (Minimal)

### Step A: Build an algorithm

```python
from rllib.model_free.baselines.policy_based.on_policy.ppo import ppo

algo = ppo(
    obs_dim=24,
    action_dim=4,
    device="cpu",
)
```

### Step B: Build and run trainer

```python
import gymnasium as gym
from rllib.model_free.common.trainers import build_trainer

def make_env():
    return gym.make("Pendulum-v1")

trainer = build_trainer(
    make_train_env=make_env,
    make_eval_env=make_env,
    algo=algo,
    total_env_steps=200_000,
    seed=42,

    enable_logger=True,
    logger_log_dir="./runs",
    logger_exp_name="ppo_pendulum",

    # transition logger (trainer-level switch)
    enable_transition_log=False,
)

trainer.train()
```

---

## 6) End-to-End Examples by Family

### 6.1 Value-based (DQN + discrete env)

```python
import gymnasium as gym
from rllib.model_free.baselines.value_based.dqn import dqn
from rllib.model_free.common.trainers import build_trainer

env = gym.make("CartPole-v1")
obs_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.n)

algo = dqn(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device="cpu",
)

trainer = build_trainer(
    make_train_env=lambda: gym.make("CartPole-v1"),
    make_eval_env=lambda: gym.make("CartPole-v1"),
    algo=algo,
    total_env_steps=150_000,
    enable_logger=True,
    logger_exp_name="dqn_cartpole",
    enable_transition_log=False,
)
trainer.train()
```

### 6.2 On-policy continuous (PPO)

```python
import gymnasium as gym
from rllib.model_free.baselines.policy_based.on_policy.ppo import ppo
from rllib.model_free.common.trainers import build_trainer

tmp = gym.make("Pendulum-v1")
obs_dim = int(tmp.observation_space.shape[0])
act_dim = int(tmp.action_space.shape[0])

algo = ppo(obs_dim=obs_dim, action_dim=act_dim, device="cpu")

trainer = build_trainer(
    make_train_env=lambda: gym.make("Pendulum-v1"),
    make_eval_env=lambda: gym.make("Pendulum-v1"),
    algo=algo,
    total_env_steps=300_000,
    enable_eval=True,
    cb_eval_every_steps=25_000,
    enable_ckpt=True,
    cb_save_every_steps=50_000,
    logger_exp_name="ppo_pendulum",
)
trainer.train()
```

### 6.3 Offline RL (CQL)

```python
from rllib.model_free.baselines.offline.cql import cql

# build your env and infer dims from observation/action spaces
# then create algo:
algo = cql(obs_dim=OBS_DIM, action_dim=ACT_DIM, device="cpu")

# feed dataset transitions via algo.on_env_step(...) or your own ingestion loop
# then train with Trainer/build_trainer similarly.
```

---

## 7) Training Infrastructure

### Core orchestration
- `common/trainers/trainer.py`: top-level lifecycle orchestration
- `common/trainers/train_loop.py`: single-env loop
- `common/trainers/train_ray.py`: Ray distributed loop
- `common/trainers/trainer_builder.py`: high-level factory entrypoint

### Policy drivers
- `common/policies/on_policy_algorithm.py`
- `common/policies/off_policy_algorithm.py`

### Update core base classes
- `common/policies/base_core.py`

### Why this split?
- Baseline code stays algorithm-focused
- Common components centralize scheduling, replay, metrics, callbacks, and logging behavior

---

## 8) Transition Logger Control (Trainer-Level)

Transition logging is controlled in `build_trainer(...)` and forwarded into `Trainer(...)`.

Relevant args:

- `enable_transition_log: bool`
- `transition_log_filename: str` (default `transitions.jsonl`)
- `transition_log_flush_every: int`

Example:

```python
trainer = build_trainer(
    ...,
    enable_transition_log=True,
    transition_log_filename="transitions_debug.jsonl",
    transition_log_flush_every=10,
)
```

When disabled, transition payloads are not written.

---

## 9) Logging and Metrics

### Scalar logger backends
- TensorBoard
- CSV
- JSONL
- W&B
- stdout

Configured through `build_trainer(...)` logger flags such as:

- `enable_logger`
- `enable_tensorboard`
- `enable_csv`
- `enable_jsonl`
- `enable_wandb`
- `enable_stdout`

### Logger paths
- trainer run directory and logger run directory are coordinated by `trainer_builder`.

---

## 10) Callback System

Callbacks are created via `common/callbacks/callback_builder.py` and plugged into trainer automatically by `build_trainer(...)`.

Typical toggles:

- `enable_eval`
- `enable_ckpt`
- `enable_best_model`
- `enable_early_stop`
- `enable_nan_guard`
- `enable_episode_stats`
- `enable_timing`
- `enable_lr_logging`
- `enable_grad_param_norm`

Representative callback parameters:

- `cb_eval_every_steps`
- `cb_save_every_steps`
- `cb_keep_last_checkpoints`
- `cb_early_stop_patience`
- `cb_best_metric_key`

---

## 11) Wrappers and Preprocessing

### Normalize wrapper
Enabled with:

- `enable_norm=True`
- related options: `norm_obs`, `norm_reward`, `norm_gamma`, `norm_epsilon`, etc.

### Atari wrapper
Enabled with:

- `enable_atari_wrapper=True`
- frame skip/stack/grayscale/resize/reward clip options

---

## 12) Replay and Buffer Features

In `common/buffers`:

- replay buffer
- prioritized replay buffer
- rollout buffer
- hindsight replay buffer

Integrated via policy driver and trainer builder options (`use_per`, `use_her`, sequence-related replay options in off-policy flow).

---

## 13) Distributed Rollout with Ray

`Trainer` supports multi-worker mode when `n_envs > 1`.

Key parameters:

- `n_envs`
- `rollout_steps_per_env`
- `sync_weights_every_updates`
- `ray_env_make_fn` (optional custom worker env factory)
- `ray_policy_spec` (optional explicit worker policy factory spec)

The codepath is in `common/trainers/train_ray.py`.

---

## 14) Checkpointing and Evaluation

Core helpers:

- `common/trainers/train_checkpoint.py`
- `common/trainers/train_eval.py`
- `common/trainers/evaluator.py`

Control through callback toggles in `build_trainer(...)`.

---

## 15) Extending the Library

Recommended pattern for adding a new algorithm:

1. Create baseline package under `baselines/.../<algo_name>/`
2. Implement:
   - `head.py`
   - `core.py`
   - `<algo_name>.py` (builder)
   - `__init__.py` exports
3. Reuse common stack:
   - policy drivers in `common/policies`
   - optimizers/schedulers in `common/optimizers`
   - trainer pipeline via `build_trainer(...)`
4. Add smoke test script (construction + update step) to validate wiring quickly.

---

## 16) Practical Import Paths

- Trainers:
  - `from rllib.model_free.common.trainers import Trainer, build_trainer`
- Loggers:
  - `from rllib.model_free.common.loggers import build_logger`
- Callbacks:
  - `from rllib.model_free.common.callbacks import build_callbacks`
- Examples:
  - `from rllib.model_free.baselines.value_based.dqn import dqn`
  - `from rllib.model_free.baselines.policy_based.on_policy.ppo import ppo`
  - `from rllib.model_free.baselines.offline.cql import cql`

---

## 17) Troubleshooting

- **Shape mismatch errors**
  - verify `obs_dim`/`action_dim` are derived from the exact wrapped env used for training
- **No training updates**
  - check off-policy warmup/update gating settings (`update_after`, `update_every`, `batch_size`, replay fill)
- **Ray worker issues**
  - verify Ray is installed and `ray_env_make_fn` is serializable
- **Unexpected transition log size**
  - disable with `enable_transition_log=False` or increase `transition_log_flush_every`
- **Missing metrics in logs**
  - ensure logger backend toggles are enabled and callback intervals are non-zero

---

## 18) Development Conventions

- Keep algorithm-specific logic inside `baselines/*`
- Keep reusable infrastructure in `common/*`
- Prefer centralized utilities/builders over duplicating logic
- Document public factories and exported symbols in module `__init__.py`
