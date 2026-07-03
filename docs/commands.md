# CLI Reference

Complete reference for the `connect4` command. `python -m connect4` is an
exact alias. All flags below are read from
[src/connect4/cli/](../src/connect4/cli/) — that code is the source of truth.

**Run everything from the repo root.** Model loading (`runs/`) and result
output (`results/`) resolve relative to the current working directory.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -e .                  # core: torch, numpy, tqdm
pip install -e ".[ui]"            # + pygame (graphical board)
pip install -e ".[dev]"           # + pytest
pip install -e ".[ui,dev]"        # everything
```

- Requires Python 3.10+.
- On platforms where pygame has no wheels (e.g. Windows ARM64),
  `pip install pygame-ce` is a drop-in replacement for the `[ui]` extra.
- CUDA users: install torch from [pytorch.org](https://pytorch.org/get-started/locally/)
  first; the trainer and RL agent use CUDA automatically when available.

`connect4 --help` is instant — heavy imports (torch, pygame) only happen
inside the subcommand you actually run.

## Subcommands

| Command | What it does |
|---|---|
| `connect4 play` | Terminal game between any two agents (or you) |
| `connect4 ui` | Pygame board (needs the `[ui]` extra) |
| `connect4 eval` | Head-to-head benchmark, alternating first player |
| `connect4 train` | Self-play RL training |
| `connect4 tournament` | The report's full round-robin (~8-10 h) |
| `connect4 experiment` | MCTS-vs-minimax scaling sweeps, parts 1-4 |

## Agent specifier grammar

`--agent1` / `--agent2` accept a spec string
(parsed by [agents/factory.py](../src/connect4/agents/factory.py)):

| Spec | Agent | Notes |
|---|---|---|
| `human` | Keyboard (terminal) or mouse (UI) input | |
| `random` | Uniform random legal move | |
| `rule` | Win > block > center heuristic | |
| `minimax` | Alpha-beta minimax, **depth 5** | |
| `minimax-<depth>` | Alpha-beta minimax at that depth | e.g. `minimax-7`; each depth step costs ~9x more time |
| `mcts` | MCTS, **500 iterations** | |
| `mcts-<iterations>` | MCTS at that iteration budget | e.g. `mcts-700`, the tournament winner |
| `rl` | Policy-value network | Loads `runs/rl_pure_selfplay_v3/best_model.pt` by default |

Suffixed specs carry their own strength; for bare `mcts`/`minimax` you can
also set it with `--iterations1/2`.

### RL model selection

For `rl` agents, the checkpoint resolves as
`runs/<--modelN>/<--checkpointN>_model.pt`, unless `--model-pathN` gives an
explicit `.pt` path (which wins). Defaults: run `rl_pure_selfplay_v3`,
checkpoint `best`. The shipped run contains `best_model.pt` only —
`--checkpoint1 final` works only for runs you trained to completion yourself.

## Shared agent flags (`play`, `ui`, `eval`)

| Flag | Default | Meaning |
|---|---|---|
| `--agent1` | `human` | Player 1 spec (see grammar above) |
| `--agent2` | `mcts` | Player 2 spec |
| `--name1`, `--name2` | agent's own default | Display-name override |
| `--iterations1`, `--iterations2` | per-type default | Strength for a **bare** `mcts`/`minimax` spec (iterations / depth). Suffixed specs like `mcts-700` carry their own number and ignore this flag |
| `--model1`, `--model2` | `rl_pure_selfplay_v3` | Run-folder name inside `runs/` for an `rl` agent |
| `--checkpoint1`, `--checkpoint2` | `best` | `best` or `final` — which `*_model.pt` inside the run folder |
| `--model-path1`, `--model-path2` | none | Explicit path to an RL `.pt` checkpoint (overrides `--model`/`--checkpoint`) |

---

## `connect4 play`

Terminal game. Extra flags on top of the shared agent flags:

| Flag | Default | Meaning |
|---|---|---|
| `--no-render` | off | Don't print the board after each move |

```bash
connect4 play                                         # you vs MCTS-500
connect4 play --agent1 human --agent2 minimax-7       # you vs depth-7 minimax
connect4 play --agent1 mcts-500 --agent2 minimax-5    # watch two AIs play
```

## `connect4 ui`

Pygame board with drop animation, hover preview, undo, and per-move timings.
Takes exactly the shared agent flags; `human` slots are controlled with the
mouse. Requires the `[ui]` extra (exits with an install hint otherwise).

```bash
connect4 ui                                           # you vs MCTS-500
connect4 ui --agent1 human --agent2 rl                # you vs the shipped RL model
connect4 ui --agent1 mcts-700 --agent2 minimax-7      # AI vs AI spectator mode
```

## `connect4 eval`

Head-to-head benchmark: alternates which agent moves first, times every move,
and prints a summary (win rates, P1/P2 splits, game lengths, per-agent
internal stats). Extra flags:

| Flag | Default | Meaning |
|---|---|---|
| `--games` | `10` | Number of games |
| `--render` | off | Print the board during games |
| `--no-print-each-game` | off | Suppress the per-game result lines |
| `--print-moves` | off | Print every move as it is chosen |

```bash
connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 20
connect4 eval --agent1 rule --agent2 rl --games 50 --no-print-each-game
connect4 eval --agent1 rl --model1 my_run --checkpoint1 final --agent2 random --games 100
```

## `connect4 train`

Self-play RL training ([training/trainer.py](../src/connect4/training/trainer.py)).
Writes to `runs/<run-name>/`: `config.json` (the exact flags used),
`training_log.json` (loss/eval curves), `best_model.pt` (saved whenever the
periodic eval score — rule + 0.5·random + 0.75·MCTS win rates — improves),
`final_model.pt` (on completion), and `checkpoints/checkpoint_ep<N>.pt`.

Run flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | `200000` | Total self-play episodes (games) |
| `--run-name` | `rl_policy` | Output folder name under `--output-dir` |
| `--output-dir` | `runs` | Where run folders go |
| `--resume` | none | Path to a `checkpoint_ep<N>.pt` to resume from |
| `--seed` | `42` | RNG seed (python/numpy/torch) |
| `--n-envs` | `512` | Parallel vectorized games per outer loop |

Optimization:

| Flag | Default | Meaning |
|---|---|---|
| `--lr` | `3e-4` | AdamW learning rate (cosine-annealed to `--min-lr`) |
| `--min-lr` | `1e-5` | Cosine annealing floor |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient-norm clip (0 disables) |
| `--batch-size` | `1024` | Replay-buffer sample size per gradient step |
| `--buffer-size` | `1000000` | Circular replay buffer capacity |
| `--updates-per-episode` | `2` | Gradient steps per episode (used only when `--updates-per-batch` is 0) |
| `--updates-per-batch` | `256` | If > 0, exactly this many gradient steps per outer loop |

Network:

| Flag | Default | Meaning |
|---|---|---|
| `--channels` | `128` | Residual trunk width |
| `--num-blocks` | `6` | Residual blocks |
| `--dropout` | `0.10` | Dropout in the policy/value heads |
| `--small-network` | off | Use the small conv net instead of the ResNet |

Exploration schedule (linear anneal by episode index):

| Flag | Default | Meaning |
|---|---|---|
| `--epsilon-start` / `--epsilon-end` | `0.3` / `0.05` | Random-move probability during self-play |
| `--epsilon-decay-episodes` | `600000` | Episodes to reach `epsilon-end` |
| `--temperature-start` / `--temperature-end` | `2.0` / `0.3` | Policy sampling temperature |
| `--temperature-decay-episodes` | `800000` | Episodes to reach `temperature-end` |

Loss weights:

| Flag | Default | Meaning |
|---|---|---|
| `--policy-weight` | `1.0` | Cross-entropy policy loss weight |
| `--value-weight` | `1.0` | MSE value loss weight |
| `--entropy-weight` | `0.05` | Entropy bonus weight |
| `--augment-mirror` / `--no-augment-mirror` | on | Mirror-augment replay data (horizontal board symmetry doubles each batch) |

Checkpointing & eval:

| Flag | Default | Meaning |
|---|---|---|
| `--snapshot-interval` | `10000` | Episodes between frozen-opponent pool snapshots |
| `--max-checkpoint-pool` | `8` | Frozen opponents kept for self-play diversity |
| `--eval-interval` | `25000` | Episodes between eval rounds vs Random/Rule/MCTS |
| `--eval-games` | `100` | Games vs Random and vs Rule per eval round |
| `--eval-games-small` | `20` | Games vs MCTS per eval round |
| `--save-interval` | `25000` | Episodes between numbered checkpoints |
| `--log-interval` | `2048` | Episodes between progress log lines |
| `--mcts-eval-iterations` | `200` | Iteration budget of the MCTS eval opponent |
| `--eval-debug` | off | Per-move debug output during eval games |

```bash
# Reproduce the shipped v3 run (RTX 2080 Ti: ~93-165 eps/s)
connect4 train --episodes 1000000 --run-name rl_pure_selfplay_v3 --n-envs 512 --updates-per-batch 128

# Short smoke run with the small network
connect4 train --episodes 20000 --run-name smoke --small-network --eval-interval 10000

# Resume an interrupted run (scheduler position is re-derived from the episode count)
connect4 train --resume runs/my_run/checkpoints/checkpoint_ep250368.pt --run-name my_run --episodes 1000000
```

## `connect4 tournament`

The full round-robin used for the report: core matchups, baselines, the RL
learning curve (auto-skips checkpoints that aren't present), minimax depth
scaling, and MCTS iteration scaling. Results are written to
`<output-dir>/tournament_results.json` after **every** matchup, so an
interrupted run keeps its data. A full run is sized to finish overnight
(~8-10 h; the recorded run took 8.2 h for 26 matchups / 1,528 games).

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | `results` | Where the JSON goes |
| `--rl-run-dir` | `runs/rl_pure_selfplay_v3` | RL run folder containing `best_model.pt` (errors out if missing) |
| `--fast-games` | `100` | Games for fast matchups (RL vs minimax, non-MCTS baselines) |
| `--mcts-games` | `20` | Games for MCTS-700 matchups (MCTS-1000/2000 get proportionally fewer) |
| `--skip-slow` | off | Skip the MCTS-2000 and minimax depth-9 matchups |
| `--quick` | off | Smoke test: 10 fast / 6 MCTS games per matchup (~15 min) |

```bash
connect4 tournament --quick               # smoke test first
connect4 tournament                       # the full ~8-10 h run
connect4 tournament --skip-slow --output-dir results_rerun
```

## `connect4 experiment`

Focused MCTS-vs-minimax scaling sweeps. Four independent parts that can run
in separate terminals simultaneously; each writes its own
`<output-dir>/mcts_vs_minimax_part<N>_<name>.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--part` | required | `1`, `2`, `3`, `4`, or `all` (sequential) |
| `--quick` | off | 6 games per matchup (10 for part 3) |
| `--output-dir` | `results` | Where the JSON goes |

| Part | Contents | Games | Approx. time |
|---|---|---|---|
| 1 | MCTS-700 vs minimax depths 3/5/7/9 | 30 each | ~3 h |
| 2 | MCTS 200/500/700/1000/1500/2000 vs Minimax-7 | 30 each | ~4 h |
| 3 | MCTS-700 vs Minimax-7 extended head-to-head | 50 | ~5 h |
| 4 | MCTS-200 vs minimax depths 3/5/7/9 | 30 each | ~2 h |

```bash
connect4 experiment --part 3              # the headline head-to-head
connect4 experiment --part 1 --quick      # fast sanity check
connect4 experiment --part all            # everything, sequentially
```

---

## Common workflows

### Play against the AI

```bash
connect4 ui --agent1 human --agent2 minimax-7     # graphical (needs [ui])
connect4 play --agent1 human --agent2 mcts-700    # terminal
```

### Benchmark two agents

```bash
connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 20 --no-print-each-game
```

First player alternates automatically; the summary includes P1/P2 win splits
and average per-move times for both agents.

### Reproduce the tournament

```bash
connect4 tournament --quick               # ~15 min sanity pass
connect4 tournament                       # full run, ~8-10 h overnight
connect4 experiment --part all            # the scaling sweeps (parts also run in parallel terminals)
```

Compare your output against the shipped [results/](../results/) JSON.

### Retrain from scratch

```bash
connect4 train --episodes 1000000 --run-name my_run --n-envs 512 --updates-per-batch 128
connect4 eval --agent1 rl --model1 my_run --agent2 rule --games 100   # then benchmark it
```

Training details and hyperparameter rationale: [training.md](training.md).

### Resume training

```bash
connect4 train --resume runs/my_run/checkpoints/checkpoint_ep250368.pt --run-name my_run --episodes 1000000
```

Pass the same `--run-name` so output keeps flowing to the same folder. The
checkpoint restores model, optimizer, episode count, and best-eval metric;
the LR scheduler position is re-derived from the episode count.

### Run tests

```bash
pip install -e ".[dev]"
pytest                                    # testpaths/addopts come from pyproject.toml
```
