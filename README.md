# `rllib/model_free`

Modular model-free reinforcement learning library with:

- value-based, policy-based, and offline baselines
- reusable common stack (buffers, networks, policies, trainers, callbacks, loggers)
- single-env and Ray multi-worker training support

This folder is designed so algorithms share the same training/runtime infrastructure while keeping each baseline implementation separate and readable.

---

## Structure

```text
model_free/
├─ baselines/
│  ├─ value_based/        # DQN family + distributional variants
│  ├─ policy_based/       # on-policy + off-policy actor(-critic) methods
│  └─ offline/            # offline RL algorithms
└─ common/
   ├─ buffers/            # replay/rollout/PER/HER
   ├─ networks/           # policy/value/Q/distribution modules
   ├─ policies/           # on-policy/off-policy drivers + core abstractions
   ├─ trainers/           # Trainer + builder + train loops (single/Ray)
   ├─ callbacks/          # eval/checkpoint/early-stop/etc.
   ├─ loggers/            # CSV/JSONL/TB/W&B/stdout + transition logger
   ├─ wrappers/           # Atari + normalization wrappers
   ├─ optimizers/         # optimizer/scheduler builders (+ KFAC/Lion)
   ├─ noises/             # exploration noise builders
   └─ utils/              # shared utility helpers
```

---

## Included Baselines

### Value-based
- `dqn`, `drqn`, `c51`, `qrdqn`, `iqn`, `fqf`, `rainbow`

### Policy-based Off-policy
- `ddpg`, `td3`, `sac`, `sac_discrete`, `tqc`, `redq`, `acer`

### Policy-based On-policy
- `a2c`, `a2c_discrete`, `ppo`, `ppo_discrete`, `trpo`, `vpg`, `vpg_discrete`, `acktr`

### Offline
- `cql`, `iql`, `td3bc`

---

## Quick Start

### 1) Build an algorithm

```python
from rllib.model_free.baselines.policy_based.on_policy.ppo import ppo

algo = ppo(
    obs_dim=24,
    action_dim=4,
    device="cpu",
)
```

### 2) Build a trainer

```python
from rllib.model_free.common.trainers import build_trainer
import gymnasium as gym

def make_env():
    return gym.make("Pendulum-v1")

trainer = build_trainer(
    make_train_env=make_env,
    make_eval_env=make_env,
    algo=algo,
    total_env_steps=200_000,
    seed=42,

    # logging
    enable_logger=True,
    logger_log_dir="./runs",
    logger_exp_name="ppo_pendulum",

    # transition logger toggle (trainer-level)
    enable_transition_log=False,  # set True to write transitions.jsonl
)

trainer.train()
```

---

## Trainer Notes

- `build_trainer(...)` is the easiest entry point for production training runs.
- Supports:
  - single-env loop (`n_envs=1`)
  - Ray rollout workers (`n_envs>1`, with optional policy/env factories)
- Transition logging is controlled at trainer construction:
  - `enable_transition_log`
  - `transition_log_filename`
  - `transition_log_flush_every`

---

## Logging and Callbacks

### Logger backends
- TensorBoard
- CSV
- JSONL
- W&B
- stdout
- transition JSONL (`common/loggers/transition_logger.py`)

### Common callbacks
- evaluation
- checkpointing
- best model saving
- early stopping
- NaN guard
- timing / LR / grad-norm logging
- Ray reporting callbacks (optional)

---

## Optional Features

- **Normalize wrapper**: observation/reward normalization + action handling
- **Atari wrapper**: frame stack/skip, preprocessing, reward clipping
- **Replay variants**:
  - uniform replay
  - PER
  - HER
  - sequence replay support for recurrent methods

---

## Practical Import Paths

- Trainers:
  - `from rllib.model_free.common.trainers import Trainer, build_trainer`
- Loggers:
  - `from rllib.model_free.common.loggers import build_logger`
- Callbacks:
  - `from rllib.model_free.common.callbacks import build_callbacks`
- Example baseline:
  - `from rllib.model_free.baselines.value_based.dqn import dqn`

---

## Development Tips

- Add new algorithms by composing:
  1. `Head` (networks/inference),
  2. `Core` (update/loss),
  3. policy driver (`OnPolicyAlgorithm` or `OffPolicyAlgorithm`),
  4. trainer wiring via `build_trainer(...)`.
- Keep algorithm-specific logic in `baselines/*` and shared logic in `common/*`.

