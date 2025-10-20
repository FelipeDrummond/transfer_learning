# Simple World Communication — Multi‑Agent RL with RLlib

Final project for a Reinforcement Learning course using PettingZoo's multi‑agent MPE environment and RLlib. The setup features cooperative "good" agents and adversaries with a designated leader that communicates discrete tokens to followers. The leader can use either a baseline MLP communication head or a BERT‑initialized semantic communication head.

## Overview

- **Environment**: PettingZoo `simple_world_comm_v3` (multi‑agent particle env with communication).
- **Algorithm**: RLlib (PPO configured in `train_rllib.py`).
- **Leader communication**:
  - Baseline: logits over a fixed vocabulary (learned end‑to‑end).
  - BERT: projects features to BERT space and scores dot‑product to fixed vocab embeddings.

## Project structure

- `main.py`: Minimal environment loop example (rendering scaffold).
- `models.py`: RLlib Torch models for good agents, follower adversaries, and two leader variants (baseline and BERT).
- `train_rllib.py`: RLlib multi‑agent training script, policy mapping, and config.
- `README.md`: This file.

## Installation

1) Create and activate a virtual environment (recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
```

2) Install dependencies.

```bash
pip install "ray[rllib]" gymnasium pettingzoo torch transformers
```

Notes:
- Torch install may vary by platform/GPU. Refer to the official PyTorch site if you need CUDA/MPS builds.
- Transformers will download the BERT weights on first use when running the BERT leader.

## Training

Key toggles and settings live in `train_rllib.py`:

- `LEADER_VOCAB`: list of communication tokens the leader can emit.
- `USE_BERT_LEADER`: set to `True` to use the BERT‑based leader, `False` for baseline MLP.
- GPU usage is auto‑detected via `torch.cuda.device_count()`; adjust as needed.

Run training:

```bash
python train_rllib.py
```

During training, the script prints aggregate and per‑policy rewards and periodically saves checkpoints.

## Models (from `models.py`)

- `GoodAgentNetwork`: MLP policy/value for cooperative agents (movement actions only).
- `FollowerAdversaryNetwork`: MLP policy/value for follower adversaries that consume observations including leader communication.
- `LeaderAdversaryNetworkBaseline`: MLP body with separate heads for movement and communication; combined action space equals `vocab_size * num_move_actions`.
- `LeaderAdversaryNetworkBERT`: MLP body + projection into BERT embedding space; communication logits via dot‑product with precomputed vocab embeddings; combined with movement logits.

## Environment quick‑view

`main.py` includes a minimal `agent_iter` loop scaffold for the PettingZoo env with rendering. It is intended as a visualization starting point and does not wire up trained RLlib policies directly. To just sanity‑check rendering with random actions, you can use a short snippet like:

```python
import numpy as np
from pettingzoo.mpe import simple_world_comm_v3

env = simple_world_comm_v3.env(render_mode="human")
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = None if term or trunc else env.action_space(agent).sample()
    env.step(action)
env.close()
```

## Troubleshooting

- Module import differences:
  - `main.py` uses `from pettingzoo.mpe import simple_world_comm_v3`.
  - `train_rllib.py` currently imports the env from `mpe2.simple_world_comm_v3`, which assumes a locally available variant. If you don't have `mpe2`, switch that import to PettingZoo's `pettingzoo.mpe.simple_world_comm_v3` or ensure your custom `mpe2` package is installed.

- Large model downloads: The BERT leader downloads weights the first time; ensure internet access or pre‑cache models via `transformers` tooling.

- Hardware: If using GPU, verify your Torch install matches your CUDA/MPS stack. Otherwise, training runs on CPU.

## Notes

- This codebase uses RLlib Torch models (`TorchModelV2`) and follows multi‑agent training with a policy mapping function in `train_rllib.py`.
- Checkpoints are saved periodically during training; see the terminal output for exact locations.
