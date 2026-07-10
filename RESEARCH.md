# Monte Carlo Agent Research Questions

Tracked alongside `benchmark/evaluate.py` (harness) and `benchmark/results.csv` (data). Findings summarized in `benchmark/MC_EVALUATION.md`. Baseline "best exact search" = `MTDfAgent` (see plan for rationale).

## Primary questions

- [ ] **Q1 — MC vs. best exact search.** How well do MC algorithms perform against `mtdf`? *Hypothesis: MC agents lose or draw more often than they win, with variance across MC variants.*
- [ ] **Q2 — Convergence with simulation count.** Does MC move-accuracy/win-rate approach exact-search optimality as `n_simulations` grows? *Hypothesis: yes, asymptotically.*
- [ ] **Q3 — Degradation with board size.** At fixed `n_simulations`, does performance drop as board size (branching factor) grows? *Hypothesis: yes, sharply.*
- [ ] **Q4 — Ideal simulation/board-size scaling.** Is there a ratio of `n_simulations` to board size (and `k`) needed to hold win-rate steady? *Hypothesis: required sims scale roughly with branching factor / depth, worse for larger `n`, better (more linear) for smaller `k`.*
- [ ] **Q5 — Search efficiency.** Which MC variant reaches near-optimal decisions with the fewest simulations? *What metric?* — move-accuracy against the `mtdf` oracle at matched `sims`, and sims-to-accuracy-threshold as the efficiency metric.

## Bonus questions

- [ ] **Q6 — Effect of `k` independent of `n`.** Does shrinking `k` (shallower games) offset the `n`-driven branching-factor penalty from Q3?
- [ ] **Q7 — First-move advantage per agent type.** Do MC agents realize the X-goes-first edge as reliably as `mtdf`, or does simulation noise erode it?
- [ ] **Q8 — Equal-wall-clock-time comparison.** Does the Q5 efficiency ranking change when compared by wall-clock time instead of raw simulation count (PUCT pays for a net forward pass per sim)?
- [ ] **Q9 — Does the untrained PUCT prior help or hurt?** `PolicyValueNet` has random, never-trained weights — is `mc_puct` actually worse than plain `mc_uct` at matched sims?
- [ ] **Q10 — Draw-rate trend.** As sims/board-size/`k` vary, does the MC agent converge toward "mostly draws" (near-optimal) or "mostly losses" (far from optimal)?
- [ ] **Q11 — Which heuristic drives `mc_informed`'s edge?** Sweep single heuristics vs. the full ensemble (`distance`, `fork`, `taxonomy`, `threat`, `window_scorer`) in `MCInformedAgent`'s rollout guidance.
