# Connect 4 AI Engine

[![CI](https://github.com/whuang214/connect-4-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/whuang214/connect-4-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Three game-AI paradigms — **minimax with alpha-beta pruning**, **Monte Carlo
Tree Search**, and a **self-play deep RL agent** (2.3M-parameter policy-value
ResNet) — implemented from scratch and compared head-to-head across a
~2,000-game tournament.

**Headline result:** MCTS-700 beat depth-7 minimax in **60% of 130 games**
(winning from both seats), while the pure self-play RL agent — despite 500k
training episodes on a custom vectorized engine — lost to every search agent,
a negative result analyzed in depth in [docs/results.md](docs/results.md).

## Quickstart

```bash
git clone https://github.com/whuang214/connect-4-engine
cd connect-4-engine
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -e ".[ui,dev]"
```

Requires Python 3.10+. The `[ui]` extra pulls pygame for the graphical board
(on platforms without pygame wheels, e.g. Windows ARM64, `pip install pygame-ce`
works as a drop-in); the core engine, agents, and training need only
torch/numpy/tqdm. CUDA users: install torch from
[pytorch.org](https://pytorch.org/get-started/locally/) first.

```bash
connect4 ui   --agent1 human --agent2 minimax-7        # graphical game
connect4 play --agent1 mcts-500 --agent2 minimax-5     # terminal game
connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 20
connect4 eval --agent1 rule --agent2 rl --games 50     # RL model ships in-repo — works out of the box
```

Run commands from the repo root — model loading (`runs/`) and result output
(`results/`) resolve relative to the current directory.
`python -m connect4 <subcommand>` works as an alternative to `connect4`.

## Commands

| Command | What it does |
|---|---|
| `connect4 play` | Terminal game between any two agents (or you) |
| `connect4 ui` | Pygame board with move timings, undo, and hover preview |
| `connect4 eval` | Head-to-head benchmark with first-player alternation and per-move timing |
| `connect4 train` | Self-play RL training (512 parallel envs, checkpointing, resume) |
| `connect4 tournament` | The full round-robin used for the report (~8h; `--quick` ≈ 15 min) |
| `connect4 experiment` | MCTS-vs-minimax scaling sweeps, parts 1–4 |

Full flag reference and workflows: [docs/commands.md](docs/commands.md).

## Agents

| Spec | Agent | Notes |
|---|---|---|
| `human` | Keyboard/mouse input | |
| `random` | Uniform random legal move | Baseline |
| `rule` | Win → block → center heuristic | Baseline |
| `minimax`, `minimax-<depth>` | Alpha-beta with windowed positional heuristic | Default depth 5 |
| `mcts`, `mcts-<iterations>` | UCT + tactical overrides + threat-aware rollouts + tree reuse | Default 500 iters; tournament winner |
| `rl` | Policy-value ResNet trained by self-play | Loads `runs/rl_pure_selfplay_v3/best_model.pt` by default |

Per-agent algorithm write-ups: [docs/agents/](docs/agents/README.md).

## Results at a glance

| Matchup | Result |
|---|---|
| MCTS-700 vs Minimax-7 | **60% MCTS** over 130 games |
| MCTS-700 vs Minimax-3/5/9 | 84% / 70% / 56% MCTS |
| Minimax-7 vs Random & Rule | 100% |
| RL vs Random | 92% |
| RL vs Rule-Based | 37% |
| RL vs any search agent | 0% (flat from 100k→500k episodes) |

Full tables, decision-time analysis, and the RL failure analysis:
[docs/results.md](docs/results.md) · original report: [docs/report.pdf](docs/report.pdf).

## Training

```bash
connect4 train --episodes 1000000 --run-name my_run --n-envs 512 --updates-per-batch 128
```

Checkpoints save to `runs/<run-name>/checkpoints/`; resume with `--resume`.
Pipeline details, hyperparameters, and the v1→v3 debugging journey (O(n)
replay buffer → circular NumPy arrays; LR-scheduler misconfiguration):
[docs/training.md](docs/training.md).

## Project structure

```
src/connect4/
├── engine.py            # OO game engine — undo stack enables in-place tree search
├── tactics.py           # shared immediate win/block detection + center ordering
├── agents/              # human, random, rule-based, minimax, MCTS, RL + factory
├── models/              # policy-value residual CNN + board encoding
├── training/            # vectorized 512-env self-play engine + trainer
├── evaluation/          # head-to-head runner + shared tournament harness
├── ui/                  # pygame board
└── cli/                 # `connect4` subcommands
tests/                   # engine/agents/training/CLI unit tests (CI-run)
docs/                    # full documentation + the original report
runs/                    # shipped tournament model + training logs
results/                 # tournament output (JSON)
```

Annotated navigation map: [docs/file-structure.md](docs/file-structure.md) ·
design rationale (incl. why there are two engines): [docs/architecture.md](docs/architecture.md).

## Tests

```bash
pytest
```

The suite cross-validates the vectorized engine against the reference engine
move-for-move and pins the two board encoders bit-identical, alongside
engine/agent/CLI unit tests. CI runs on every push.

## Credits

CS5100 (Foundations of Artificial Intelligence) final project — Will Huang,
Joseph Winterlich, Soham Santra. MIT licensed.
