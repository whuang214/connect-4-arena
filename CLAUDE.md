# CLAUDE.md

Connect 4 AI engine: minimax, MCTS, and self-play RL agents compared in a
tournament framework. src-layout Python package, installed editable.

## Orientation

- **Navigation map (start here):** [docs/file-structure.md](docs/file-structure.md)
  — every module, its key symbols, and "where to look when changing X".
- **CLI reference:** [docs/commands.md](docs/commands.md)
- **Design rationale:** [docs/architecture.md](docs/architecture.md) — notably
  the two-engine design: `engine.py` (OO, undo stack, used by tree search) vs
  `training/vec_engine.py` (batched NumPy, used by RL self-play). Their board
  encoders must stay bit-identical; tests pin this.

## Commands

```bash
pip install -e ".[ui,dev]"     # setup (pygame optional; pygame-ce OK on ARM64)
pytest                          # run the test suite
connect4 <play|ui|eval|train|tournament|experiment> ...
```

Run everything from the repo root — `runs/` and `results/` paths are
CWD-relative.

## Conventions

- Board: row 0 = top; players are 1/2; `EMPTY = 0`. Drop row = `ROWS-1-height`.
- Network I/O: 4-channel current-player-perspective encoding
  (`models/policy_value_network.py:encode_board`); value head is tanh [-1, 1].
- `connect4.agents` re-exports lazily (PEP 562): torch loads only when the RL
  agent is used; pygame only inside `connect4 ui`. Keep `cli/*.py` module level
  stdlib-only so `connect4 --help` stays instant.
- Shared tactical helpers (immediate win/block, center ordering) live in
  `connect4/tactics.py` — don't re-implement them in agents.
- `runs/**/*.pt` is gitignored except the shipped
  `runs/rl_pure_selfplay_v3/best_model.pt`. Never commit new checkpoints.
