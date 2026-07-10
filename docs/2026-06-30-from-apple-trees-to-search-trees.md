---
title: "From Apple Trees to Search Trees: Optimising Classical AI to Overcome TicTacToe"
subtitle: "How seven decades of game-tree research turned a children's game into a proving ground for AI"
date: 2026-06-30
category: Algorithms
abstract: "We left TicTacToe broken. Now we fix it — by treating it as a search problem. This post walks through the classical AI search toolkit: from MiniMax to Best Node Search, each algorithm a sharper answer to the same question of how to play perfectly without drowning in possibilities."
comments: true
---

*This post was proofread with the assistance of AI.*

---

Last time, I handed a 4×4 TicTacToe board to children at a volunteering centre, changed the win condition to 3 in a row, and watched a familiar game become genuinely hard. I also showed why that hardness is not superficial: the number of reachable game states grows exponentially, early wins make closed-form analysis intractable, and even a simple question like "who wins under random play?" requires a full recursive search.

That is where AI begins: with Classical AI. The field's answer is deceptively simple: **treat every problem as a search problem**. Define the game formally, build a tree of future possibilities, and search it intelligently. This post traces how that idea evolved, from the naïve depth-first search a first-year student might attempt, through seven decades of refinements, to the algorithms that power modern chess engines.

---

## Formalising the Game: The START Framework

Before we can search, we need a language for the problem. In AI, every search problem fits a five-part skeleton I find easiest to remember as **START**: **S**tate, **T**ransition, **A**ction, **R**eward, **T**erminal.

**State** $s$ is a mathematical snapshot of the game at a moment in time. For a 3×3 TicTacToe board, one natural representation is a $3 \times 3$ matrix where each cell holds `X`, `O`, or `null`.

```
Board               Matrix representation
 X | O | .         [["X",   "O",  null],
---+---+---   →     [null,  "X",  null],
 . | X | .          [null,  null, "O" ]]
---+---+---
 . | . | O
```

State design is an art. Include too little and the agent cannot distinguish situations that call for different moves. Include too much, the clock time, the room temperature, a player's hydration level, and the search space balloons needlessly. The *state space* (the set of all possible states $\mathcal{S}$) is always finite for TicTacToe, but its size depends entirely on what the state tracks. In AI terms, a game where both players see the entire board is called *fully observable*; one where players see only a portion (as in poker) is *partially observable*. TicTacToe is fully observable, which simplifies our state considerably.

**Transition** $T$ describes how one state becomes another: $s \xrightarrow{T} s'$. For TicTacToe, a transition places exactly one mark on one empty cell. A useful sanity check: the number of `null` cells in $s'$ must always be exactly one fewer than in $s$. Simple invariants like this catch implementation bugs before they compound.

**Action** $A(s)$ is the set of moves available from state $s$ — the empty cells, in TicTacToe. Not every action is legal in every state, so $A(s)$ shrinks as the game progresses. The full set of all actions across all states is the *action space*.

**Reward** $R(s, a)$ is optional but important. It assigns a numerical value to states or action sequences so the agent can prefer some paths over others. For TicTacToe, the simplest choice is a terminal score: $+1$ for a win, $-1$ for a loss, $0$ for a draw. We will revisit reward design below — it is easier to get wrong than it looks.

**Terminal** defines when the search stops: any state where a player has formed $k$ consecutive marks in a line, or every cell is filled with no winner.

---

## The Search Tree

With the game formalised, the search becomes a tree. The root is the empty board. Each edge is a legal action. Each child node is the resulting board. The tree terminates at terminal states.

```
                    [empty board]
                  /       |        \
           [X:top-L]  [X:centre]  [X:top-R]  ...  (9 branches)
           /       \
     [O:top-M]  [O:centre]  ...               (8 branches each)
```

Let $b$ be the branching factor at any node (the number of legal moves) and $d$ the maximum depth (the longest possible game). The total number of leaf nodes is at most $O(b^d)$. For a 3×3 game, $b$ starts at 9 and $d = 9$, so the tree has at most $9! = 362{,}880$ leaves — manageable on any modern machine. For a 5×5 board, the same calculation gives $25! \approx 1.6 \times 10^{25}$. No machine alive will enumerate that.

This is the wall every exact search hits. The tree exists in principle; it is simply too large to traverse completely. Every algorithm in the rest of this post is an attempt to search *less* of it while still arriving at the *same* answer.

---

## The Algorithm Progression

The following seven algorithms are not a menu of alternatives. They are a chain: each one identifies the precise remaining weakness of its predecessor and fixes it.

### 1. MiniMax

![MiniMax game tree — squares maximise, circles minimise, values propagate from leaves to root](https://upload.wikimedia.org/wikipedia/commons/6/6f/Minimax.svg)
*A MiniMax game tree. Square nodes maximise; circle nodes minimise. Terminal values ($+1$, $0$, $-1$) propagate upward to yield the root's optimal value.*

MiniMax adapts depth-first search for two-player zero-sum games. One player (the *maximiser*, X) drives the terminal score upward; the other (the *minimiser*, O) pushes it down. Terminal nodes receive scores of $+1$, $-1$, or $0$. Internal nodes take the max or min of their children depending on whose turn it is.

$$
v(s) = \begin{cases}
R(s) & \text{if } s \text{ is terminal} \\
\max_{a \in A(s)}\, v(T(s, a)) & \text{if maximiser's turn} \\
\min_{a \in A(s)}\, v(T(s, a)) & \text{if minimiser's turn}
\end{cases}
$$

The result is optimal play against a perfectly rational opponent. The cost: MiniMax visits every node in the tree, $O(b^d)$ nodes in total. It works for 3×3 TicTacToe. It drowns on anything larger.

### 2. Alpha-Beta Pruning

![Alpha-Beta pruning — crossed-out branches are safely ignored](https://upload.wikimedia.org/wikipedia/commons/9/91/AB_pruning.svg)
*Alpha-Beta pruning in action. The crossed-out subtrees cannot change the root value regardless of their contents and are safely skipped.*

Alpha-Beta pruning keeps MiniMax's exact guarantees while ignoring subtrees that cannot influence the outcome. It tracks two bounds:

- $\alpha$ — the best score the maximiser has secured so far (a lower bound on what the maximiser will accept)
- $\beta$ — the best score the minimiser has secured so far (an upper bound on what the minimiser will accept)

Whenever $\beta \leq \alpha$, the current branch is *cut off*: the minimiser would never allow a result this good for the maximiser, so there is no point searching further.

With *perfect move ordering* — best moves explored first — Alpha-Beta reduces the effective tree size from $O(b^d)$ to $O(b^{d/2})$. The practical implication: the same computational budget now reaches twice the depth, or equivalently, a search that would take 100 hours takes only 10. In practice, move ordering is imperfect, and performance sits somewhere between the two bounds. Still, even average-case pruning yields dramatic savings.

### 3. NegaMax

NegaMax notices that MiniMax's maximiser and minimiser are doing the same thing in opposite directions. If we negate the returned score each time the active player switches, both players become maximisers of their own perspective. The recursion collapses to a single function:

$$
v(s) = \max_{a \in A(s)}\bigl(-v(T(s, a))\bigr)
$$

The search tree explored is identical to MiniMax with Alpha-Beta. NegaMax is purely a *cleaner implementation*: fewer variables, one recursive case instead of two, less code to audit for bugs. In a domain where an off-by-one error in score propagation produces subtly wrong play, simplicity has real value.

### 4. Scaled Rewards: A Warning About Heuristics

An appealing optimisation is to reward faster wins more than slower ones — score a five-move win as $+1.0$ and a seven-move win as $+0.8$, so the search gravitates toward quick victories. The intuition is sound. The consequence is not.

In plain MiniMax with a reward space of $\{-1, 0, +1\}$, the moment the maximiser finds a $+1$ terminal it knows the absolute best outcome has been achieved. No further search is needed at that node. With scaled rewards, a $+0.8$ today might coexist with an undiscovered $+1.0$ elsewhere. The algorithm can no longer stop early — it must confirm it has the *shortest* win, not merely *a* win. A heuristic designed to shrink the search quietly changed the question from "find any win" to "find the fastest win," and the latter is strictly harder.

The lesson generalises: domain-aware heuristics can speed up search in one dimension while slowing it down in another. Measure before assuming.

### 5. NegaScout

NegaScout exploits a pattern that emerges with good move ordering: if the first move explored at any node really is the best, then every subsequent sibling only needs to be *confirmed worse* — we do not need its exact score. Confirming inferiority is cheaper than measuring precisely.

To do this, NegaScout searches sibling nodes with a *null window* $[\alpha,\, \alpha + 1]$ instead of the full window $[\alpha, \beta]$.

```
Standard window:   [α ─────────────────── β]
Null window:       [α ─ α+1]
```

A null-window search is fast because almost everything gets pruned at once. If the result falls inside the window, the sibling is confirmed worse — search moves on. If it exceeds the window, the initial assumption was wrong: this sibling is actually better, and a full re-search is needed.

NegaScout is thus fast when the first move truly is the best, and degrades gracefully when it is not. Its efficiency is **entirely dependent on move ordering**: a bad first move triggers a full re-search plus the overhead of the null-window attempt.

### 6. MTD(f)

MTD(f), published by Aske Plaat and colleagues in 1994, takes the null-window idea to its logical conclusion: use *only* null-window searches, every time, for every node. The algorithm works by binary search over the true minimax value.

Starting from a guess $f$ (ideally close to the true value), it calls NegaMax with the window $[f, f+1]$. Each call returns either a lower bound or an upper bound on the true value. Successive calls narrow the interval until the bounds meet.

$$
\text{repeat until } \ell = u:\quad \text{NegaMax}([\,f,\, f+1\,]) \to \text{new bound; update } \ell, u, f
$$

Because every call is a null-window search, every call benefits from maximum pruning. The cost is that multiple passes revisit parts of the tree. A *transposition table* — a cache of previously seen positions — is therefore not optional; without it, repeated searches undo all the savings. MTD(f) consistently outperforms NegaScout by around 5–10% on chess, checkers, and Othello benchmarks, trading the simplicity of NegaScout's single pass for the deeper pruning of repeated null-window calls.

### 7. Best Node Search

Best Node Search (BNS, 2011) reframes the goal entirely. Rather than computing the *value* of the optimal move, it only needs to identify *which move is optimal*. That is a strictly weaker question — and weaker questions can sometimes be answered more efficiently.

BNS iteratively guesses a threshold and counts how many moves score above and below it, adjusting the threshold until only one candidate remains above it.

```
Guess threshold = 0.4 → 11 moves worse,  9 better
Guess threshold = 0.6 → 19 moves worse,  1 better  ← one candidate remains
Result: best move found. Exact value unknown, but in (0.4, 0.6).
```

Once only one move exceeds the threshold, that move is optimal by elimination — no further evaluation needed. BNS achieves strong empirical performance in large search spaces precisely because it stops asking "how good is this?" the moment the contest between top candidates is settled. Like MTD(f), it benefits significantly from a good initial threshold and a transposition table.

---

## Iterative Deepening: The Crosscutting Technique

Every algorithm above benefits from two things it cannot easily provide for itself: a good initial value (for MTD(f) and BNS) and a good move ordering (for NegaScout and Alpha-Beta). Iterative deepening supplies both at low cost.

The idea: run a complete depth-limited search to depth 1, take the best move found, then search to depth 2 using that result, then depth 3, and so on. Revisiting earlier depths looks wasteful, but the overhead is bounded. Because the final depth dominates the total node count — exponentially more nodes live at depth $d$ than at all shallower depths combined — the overhead factor is only $\frac{b}{b-1}$, a constant independent of depth.

The payoff is that the best move found at depth $d$ becomes the first move explored at depth $d+1$, giving NegaScout exactly the move-ordering it needs. The minimax value from depth $d$ becomes MTD(f)'s and BNS's initial guess at depth $d+1$ — not perfect, but far better than no information at all.

Iterative deepening does not make any single algorithm asymptotically faster. It makes the whole family of algorithms work better *together*.

---

## What I Took Away

**Incremental progress matters.** MiniMax feels almost childishly simple: explore every move, propagate values upward. But without it, there is no Alpha-Beta. Without Alpha-Beta, there is no NegaScout. Without NegaScout, there is no MTD(f). Each algorithm is a targeted fix to a precisely identified weakness — not a replacement of what came before, but a sharpening of it. Progress in this field did not come from discarding old ideas but from asking "what specifically is still slow?" and fixing exactly that.

**Loosening assumptions is not imprecision.** NegaScout assumes the first move is best. BNS assumes you only need to identify *which* move wins, not by how much. Both assumptions are sometimes wrong. Both algorithms degrade gracefully when the assumption fails, and when it holds, the savings are large. An assumption with a known failure mode is not a weakness; it is an explicit design decision.

**Heuristics change the question.** The Scaled Rewards example is a case study in unintended consequences. The heuristic was designed to shrink the search. It accomplished that — but only by turning "find any win" into "find the fastest win." The question changed without anyone deciding to change it. The most dangerous optimisations are the ones that feel obviously correct.

---

## Food for Thought

1. Alpha-Beta pruning achieves $O(b^{d/2})$ under *perfect* move ordering. In practice, how close can heuristics like killer moves or history tables get to this ideal, and is the remaining gap worth measuring?

2. MTD(f) and BNS both depend on transposition tables. What happens to their performance in TicTacToe variants where symmetry hashing reduces very few repeated states — does the transposition table cost more in memory overhead than it saves in re-search?

3. All of the algorithms here find *optimal* play. In large-board TicTacToe, optimal play often means a guaranteed draw. Is a perfectly optimal draw solver useful in any sense a human player would actually value — or does the real challenge lie elsewhere?

---

*In the next post, I will bring these algorithms out of theory and into benchmarks — running MiniMax, Alpha-Beta, NegaScout, and MTD(f) on an actual $n \times n$ TicTacToe engine, and measuring exactly where each one runs out of breath.*
