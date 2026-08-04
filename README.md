# TicTacToe at scale

Generalised TicTacToe: an `n x n` board where a player wins by placing `k`
marks in a row. The childhood game is `n = 3, k = 3`. Raise `n` and the game
stops being solvable — this repo is a collection of game-playing algorithms
that shows exactly where that happens, and what you can do about it.

Every agent implements one interface, `act(state) -> (row, col)`, so any two
can be pointed at the same board and compared directly.

## Running a game

Two agents of the same algorithm play each other from an empty board:

```bash
python -m tictactoe.cli -n 3 -k 3 -agt mtdf
python -m tictactoe.cli -n 5 -k 4 -agt mc_uct -v      # -v prints each move
```

Output reports the winner, how many distinct board states were visited, and
wall-clock time — the two numbers that matter when comparing algorithms.

Only `numpy` is required, and only by the PUCT agent's network.

## The agents

**Exact search.** Explores the game tree to a terminal position and plays
optimally. Correct by construction, and unusable once the tree gets big.

| `-agt` | Class | What it does |
|---|---|---|
| `random` | `RandomAgent` | Uniform random legal move — the floor to beat |
| `minimax` | `MinimaxAgent` | Plain minimax, no pruning |
| `minimax_ab` | `MinimaxAlphaBetaAgent` | Minimax with alpha-beta pruning |
| `minimax_rewards_ab` | `MinimaxRewardsAlphaBetaAgent` | Alpha-beta that prefers faster wins |
| `negamax` | `NegamaxAgent` | Negamax + alpha-beta, one perspective-flipped code path |
| `negascout` | `NegascoutAgent` | Negamax with null-window re-search |
| `mtdf` | `MTDfAgent` | MTD(f): repeated null-window search over a transposition table |
| `mtdf_id` | `MTDfIDAgent` | MTD(f) with iterative deepening; heuristic evaluation at the leaves |
| `bns` | `BNSAgent` | Best Node Search: binary search on the game value |
| `bns_id` | `BNSIDAgent` | Best Node Search with iterative deepening |

**Monte Carlo tree search.** Gives up the optimality guarantee and estimates
move values by playing random games. Keeps working on boards exact search
cannot touch.

| `-agt` | Class | What it does |
|---|---|---|
| `mc_pure` | `MCPureAgent` | Uniform-random selection, no tree policy |
| `mc_uct` | `MCUCTAgent` | UCT: the standard `w/n + c*sqrt(ln N / n)` bandit rule |
| `mc_uct_forced` | `MCUCTForcedAgent` | UCT whose rollouts take a win or block a loss when one exists |
| `mc_informed` | `MCInformedAgent` | Heavy playouts guided by the heuristic ensemble |
| `mc_rave` | `MCRaveAgent` | RAVE/AMAF: shares move statistics across sibling nodes |
| `mc_puct` | `MCPUCTAgent` | PUCT with a policy/value network prior (untrained — a demonstration of the mechanism, not a strong player) |

## Layout

```
tictactoe/
├── cli.py         game runner and the agent registry above
├── agents/        one file per algorithm; all subclass BaseAgent
├── core/          Board, State (apply/undo + Zobrist hash), transposition table
└── heuristics/    position evaluators used by the *_id and mc_informed agents
tests/             pytest suite
docs/              write-ups
```

`State.apply()` / `State.undo()` mutate one board in place rather than copying,
which is what makes deep search affordable. `TTState` adds symmetry-aware
hashing on top and is used by the transposition-table agents.

## Tests

```bash
pytest tests/ -v
ruff check tictactoe/ tests/
```

## Write-ups

`docs/` holds the articles this code was written for, in order:

1. **I Ruined TicTacToe for My Children (and for Math)** — why the game breaks
   as `n` grows.
2. **From Apple Trees to Search Trees** — building exact search up from
   minimax to MTD(f).
3. **I Brought a Casino to a Children's Game** — MCTS from first principles,
   and where its ceiling actually sits.
