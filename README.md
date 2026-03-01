# `rllib`

Lightweight reinforcement learning repository organized by algorithm family and
infrastructure maturity.

---

## Repository layout

```text
rllib/
├─ classics/      # classic/tabular RL + toy environments
├─ model_free/    # model-free deep RL baselines + common training stack
└─ model_based/   # placeholder for future model-based RL implementations
```

---

## Module status

### `classics/`

Implemented.

Includes:

- bandits
- dynamic programming
- Monte Carlo prediction/control
- temporal-difference prediction/control
- tabular policy-gradient methods
- model-based tabular control (e.g., Dyna-Q, prioritized sweeping)
- MCTS
- toy environments under `classics/toy_env`

See:

- [classics/README.md](/home/jyhong/projects/rllib/classics/README.md)

### `model_free/`

Implemented.

Includes:

- value-based baselines (DQN family, distributional variants, recurrent variants)
- on-policy and off-policy policy-based baselines
- offline RL baselines
- shared trainer/callback/logger/buffer/network infrastructure

See:

- [model_free/README.md](/home/jyhong/projects/rllib/model_free/README.md)

### `model_based/`

Not implemented yet.

- The directory exists as a planned location for future model-based RL code.
- There is currently no public API, baseline implementation, or training stack
  in `model_based/`.

---

## Roadmap (high-level)

1. Add initial `model_based` module structure and public API surface.
2. Add first baseline implementation (with planning/world-model core).
3. Add integration examples mirroring `model_free` trainer ergonomics.
4. Add dedicated README and usage docs for `model_based`.

