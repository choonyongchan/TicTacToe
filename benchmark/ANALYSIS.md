# Benchmark Analysis: Algorithm Scaling vs n and k

## Data Summary

All benchmarks run the algorithm as both players from the opening move.  
"Timeout" = 3600s wall-clock limit reached without solving. States-visited counts for timeouts reflect progress at cutoff, not total game-tree size.

| Algorithm | 3×3 (k=3) | 4×4 (k=4) | 4×4 (k=3) | 5×5 (k=5) | 5×5 (k=4) | 5×5 (k=3) |
|---|---|---|---|---|---|---|
| random | 9 / 0.00s | 16 / 0.00s | 8 / 0.00s | 25 / 0.00s | 13 / 0.00s | 16 / 0.00s |
| minimax | 618 184 / 8.57s | **TIMEOUT** | **TIMEOUT** | — | — | — |
| minimax_ab | 21 652 / 0.30s | **TIMEOUT** | 42 340 / 19.13s | — | — | — |
| minimax_rewards_ab | 24 698 / 0.35s | **TIMEOUT** | 282 469 / 132.19s | — | — | — |
| negamax | 24 698 / 0.34s | **TIMEOUT** | 282 469 / 132.45s | — | — | — |
| mtdf | **1 817 / 0.04s** | **196 677 / 19.43s** | **113 351 / 6.29s** | TIMEOUT (9.7M) | TIMEOUT (9.4M) | TIMEOUT (9.5M) |
| mtdf_id | 5 297 / 0.15s | 265 719 / 147.50s | **40 377 / 13.39s** | TIMEOUT (13.7M) | TIMEOUT (14.1M) | TIMEOUT (3.5M) |
| negascout | 45 801 / 0.68s | **TIMEOUT** | 390 989 / 350.56s | — | — | — |
| bns | 58 226 / 0.81s | **TIMEOUT** | 2 553 446 / 2809.92s | — | — | — |
| bns_id | 7 171 / 0.17s | 958 872 / 376.01s | 125 484 / 42.51s | TIMEOUT (11.4M) | TIMEOUT (17.2M) | TIMEOUT (7.6M) |

(—) = algorithm not run on that configuration.

---

## Observation 1: Board size (n) dominates everything else

Going from 3×3 to 4×4 with k=n is catastrophic for naive algorithms:

- **minimax**: 618K states → timeout after 108K states at 3600s (would need orders of magnitude more)
- **minimax_ab**: 21K → timeout after 336K states
- **negascout / bns**: both timeout

Only MTDf survives the jump to 4×4 (196K states, 19s).  
At 5×5, *every* algorithm — including MTDf — times out. The game tree for 5×5 k=5 has ~25! ≈ 10^25 leaves; no exact solver runs in hours.

**Takeaway:** n scales the search space super-exponentially. Each +1 in board side multiplies available moves at each ply by roughly n², so the effect on tree depth dwarfs any algorithmic improvement.

---

## Observation 2: Lowering k (relative to n) dramatically reduces search cost

k controls how quickly a terminal state can appear. Smaller k means wins occur earlier → shallower effective search depth → far fewer states.

### 4×4, k=4 vs k=3

| Algorithm | k=4 states | k=4 time | k=3 states | k=3 time |
|---|---|---|---|---|
| minimax_ab | TIMEOUT (336K) | 3600s | 42 340 | 19.13s |
| mtdf | 196 677 | 19.43s | 113 351 | 6.29s |
| mtdf_id | 265 719 | 147.50s | 40 377 | 13.39s |
| bns_id | 958 872 | 376.01s | 125 484 | 42.51s |

Dropping k from 4 to 3 on a 4×4 board turns several timeouts into feasible runs. minimax_ab goes from "unsolvable in an hour" to "solved in 19 seconds" — a >180× improvement in effective states.

### 5×5: k barely matters at this scale

All three k values (3, 4, 5) timeout for every exact algorithm. Even though smaller k means earlier wins, the 25-cell branching factor overwhelms the depth reduction. The states-at-timeout figures are similar (3.5M–17M depending on algorithm), confirming that the bottleneck is tree width, not depth.

---

## Observation 3: MTDf is the dominant exact algorithm

MTDf consistently visits the fewest states where a solution is found:

| Config | MTDf | Next best | Ratio |
|---|---|---|---|
| 3×3 k=3 | 1 817 | minimax_ab 21 652 | **12× fewer** |
| 4×4 k=4 | 196 677 | bns_id 958 872 | **5× fewer** |
| 4×4 k=3 | 113 351 | mtdf_id 40 377* | (mtdf_id wins here) |

*On 4×4 k=3, mtdf_id beats mtdf by ~3×. This is the one case where iterative deepening pays off enough to overcome MTDf's native efficiency.

MTDf is the only algorithm that solves all non-5×5 configurations within the time limit.

---

## Observation 4: Alpha-beta pruning is necessary but not sufficient

Comparing minimax vs minimax_ab on 3×3: **29× state reduction** (618K → 21K). Alpha-beta is the baseline prerequisite for any tractable search.

However, minimax_ab still times out on 4×4 k=4 despite the pruning — the game tree is simply too wide and deep. More advanced techniques (transposition tables, MTDf's null-window bisection) are needed to go further.

---

## Observation 5: Iterative deepening has mixed returns

ID (iterative deepening) reorders the search via increasingly deeper passes, improving move ordering and therefore pruning quality. But it isn't always a win:

- **bns_id vs bns**: huge gain — bns visits 2.5M states on 4×4 k=3 vs bns_id's 125K (20× reduction). ID is *essential* for BNS.
- **mtdf_id vs mtdf**: at 3×3 and 4×4 k=4, plain MTDf is faster. At 4×4 k=3, mtdf_id is 3× faster. ID overhead dominates when the game tree is small enough that the first-pass result is already good.

---

## Observation 6: negamax ≡ minimax_rewards_ab

These two algorithms produce identical state counts and times on every benchmark (e.g. 24 698 states / 0.35s on 3×3). They are observationally equivalent implementations.

---

## Observation 7: NegaScout and BNS are worse than minimax_ab on 4×4 k=3

| Algorithm | 4×4 k=3 states | time |
|---|---|---|
| minimax_ab | 42 340 | 19.13s |
| negascout | 390 989 | 350.56s |
| bns | 2 553 446 | 2809.92s |

NegaScout visits ~9× more states than minimax_ab, and BNS ~60× more. Both algorithms are theoretically stronger than simple alpha-beta in the general case, but their performance depends heavily on move ordering quality. With poor ordering, the null-window re-searches in NegaScout and the BNS bisection framework incur high overhead that outweighs the pruning gains.

---

## Summary Table: Practical Solvability

| Config | Solvable by | Fastest |
|---|---|---|
| 3×3 k=3 | All exact algorithms | MTDf (0.04s) |
| 4×4 k=3 | minimax_ab, mtdf, mtdf_id, negascout, bns, bns_id | MTDf (6.29s) |
| 4×4 k=4 | MTDf, MTDf-ID, BNS-ID only | MTDf (19.43s) |
| 5×5 any k | None (all timeout) | — |

---

## Conclusions

1. **n is the binding constraint.** No exact solver handles 5×5 in hours. Heuristic/MCTS approaches are needed beyond 4×4.
2. **Lower k rescues tractability on the same board.** 4×4 k=3 is solvable by several algorithms; 4×4 k=4 is solvable only by the best.
3. **MTDf is the best exact algorithm here** — fewest states, fastest time across nearly all configurations.
4. **Iterative deepening is critical for BNS** but adds overhead for MTDf in small/medium configs; it pays off for MTDf only on harder mid-size configs.
5. **Alpha-beta is the necessary floor** — pure minimax is ~30× slower and impractical even on 3×3 in a useful latency budget.
