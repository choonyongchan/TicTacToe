---
title: "I Ruined TicTacToe for My Children, and for Math 😭"
subtitle: "What a children's game teaches us about AI, and why real-world AI hits combinatorial walls"
date: 2026-06-17
category: Algorithms
abstract: "A small rule change to TicTacToe turns a childhood game into a lesson on recursion, search trees, state explosion, and why AI systems often need more than brute force."
comments: true
---

*This post was proofread with the assistance of AI.*

---

All I did was draw one extra row and one extra column. That was enough to ruin TicTacToe.

The children at my volunteering centre love the normal 3x3 game. It is quick, familiar, and perfect for filling a spare couple of minutes with a primary school child. Then I gave them a 4x4 board. At first, they treated it like the same game with more space. After a few rounds, they noticed something annoying; it was much easier to block, much harder to finish, and the old tricks no longer worked cleanly.

So I made it worse. I kept the board at 4x4, but changed the win condition to 3 in a row.

That tiny rule change opened the floodgates. Suddenly every corner mattered. A harmless-looking move could threaten a row, a column, and a diagonal. The children got louder, more careful, and more competitive. I had turned a childhood game into a small mathematical headache.

And that was the point.

## A Small Game With Serious AI Bones

TicTacToe is useful because it is simple enough to hold in your head, but rich enough to expose real ideas in Classical AI. 

So what do we understand about TicTacToe as an AI environment? It is finite; the board eventually fills. It is deterministic; placing a mark has no randomness or chance attached to it. It is a perfect-information game; both players see the whole board at all times. It is adversarial; every good move for one player is bad news for the other.

That combination makes TicTacToe a clean playground for AI as search.

In standard 3x3 TicTacToe, the rules are fixed. X starts. Players alternate. Whoever gets 3 in a row wins. If the board fills without a winner, the game is a draw. Exhaustive search reveals (to no one's surprise) that under perfect play, neither player can force a win, so the result is a draw.

But once we generalise the game, the neat little toy starts to misbehave.

Let the board be `n x n`. Let `k` be the number of consecutive marks needed to win, where `k <= n`. The childhood version is just one setting: `n = 3, k = 3`. A 4x4 game where you need the full row is `n = 4, k = 4`. The more chaotic version I gave the children is `n = 4, k = 3`.

That notation looks innocent. It is not.

Here is the difference in one position.

```text
X X . .
O . . .
. . . .
. . . .
```

In `n = 4, k = 4`, X has only made a start. Two X marks are not close to winning yet. In `n = 4, k = 3`, X can win immediately by playing the third square in the top row:

```text
X X X .
O . . .
. . . .
. . . .
```

The board size stayed the same. The meaning of danger changed.

## Computing TicTacToe Requires Recursion

Suppose both players are young children playing without hints or strategy. At every turn, the current player chooses one legal empty square uniformly at random.

Here is the question: what is the probability that X wins?

For 3x3 TicTacToe, it is tempting to count board patterns directly. That temptation fades fast. To know whether X wins, we cannot only inspect the final board. We also need to know whether O would have won earlier, because TicTacToe stops the moment someone completes a line. A board that looks possible as a final arrangement may never appear in a real game, since the game may have ended two moves before.

There is no simple one-pass shortcut that respects turn order, illegal states, and early stopping. That pushes us into recursion.

Let `P_X(s)` be the probability that X eventually wins from state `s`. If `s` is already a terminal X win, `P_X(s) = 1`. If `s` is an O win or a draw, `P_X(s) = 0`. Otherwise, the current player has some set of legal moves, and random play averages over the child states:

```text
P_X(s) = (1 / n_s) * sum over child states s' of P_X(s')
```

where `s'` represents each child state after each legal move, and `n_s` is the number of legal moves from state `s`.

This is a tiny equation with a nasty implication. To evaluate the root, the empty board, we must evaluate the children. To evaluate the children, we evaluate their children. The game becomes a search tree.

For example, from an empty 3x3 board, random X has 9 possible first moves. After X chooses one, random O has 8 possible replies. After that, X has 7. Even before we ask who is playing well, the tree begins to branch:

```text
empty board
  -> X in top-left
    -> O in top-middle
    -> O in top-right
    -> ...
  -> X in center
    -> O in top-left
    -> O in top-middle
    -> ...
```

Recursion is just the clean way to say: solve each smaller board, then average the answers back up.

For the normal 3x3 board, this is still manageable. If both players choose uniformly random legal moves, X wins about `58.49%` of games, O wins about `28.81%`, and the game draws about `12.70%`.

That first-player advantage is not magic. X always moves first, so X gets the first chance to complete the fifth move of the game, the earliest possible winning point in 3x3 TicTacToe. Random O does not defend well enough to erase that advantage.

## Small Things Add Up, Very Quickly

A 3x3 board has `3^9 = 19,683` raw assignments if every square can be empty, X, or O. That number is already larger than most people expect, but it still fits comfortably on a slide.

The legal game is smaller in one sense and larger in another.

It is smaller because many raw assignments are illegal. X and O must alternate turns. The game stops early when someone wins. If we count reachable board states with early stopping, standard TicTacToe has `5,478` states.

It is larger because search algorithms care about paths, not only board snapshots. The same board can be reached through different move orders. Standard 3x3 TicTacToe has `255,168` terminal game sequences.

Now move from 3x3 to 4x4.

The raw board assignments jump from `3^9 = 19,683` to `3^16 = 43,046,721`. If we ignore early wins and simply count full move orders, a 4x4 board has:

```text
16! = 20,922,789,888,000
```

That is more than twenty trillion full-length move orders.

Of course, real games may stop early. That helps. It also makes the mathematics uglier, because early stopping depends on the exact sequence of previous moves. A win is not just a property of how many marks exist; it is a property of where they were placed and whether the game should have ended earlier. It is this property that makes finding a closed-form property intractable.

This is the first lesson algorithm learners should take seriously: small rule systems can produce huge search spaces without looking complicated. Think of the Travelling Salesman Problem or cryptography.

## The Value of `k` Changes the Strategy of the Whole Game

The number of winning lines on an `n x n` board with `k` in a row is:

```text
2n(n - k + 1) + 2(n - k + 1)^2
```

The first term counts horizontal and vertical windows. The second counts the two diagonal directions.

For standard 3x3 TicTacToe, `n = 3, k = 3`, so there are `8` winning lines. That matches what we learn as children: three rows, three columns, two diagonals.

For 4x4 with `k = 4`, there are only `10` winning lines. Four rows, four columns, two long diagonals. The board has more space, but each win still needs a full-length line. Blocking is cheap: one opponent mark ruins an entire candidate line.

For 4x4 with `k = 3`, the count jumps to `24` winning lines. The board did not change. The win condition did.

On a 10x10 board, the contrast is sharper. With `k = 10`, there are `22` winning lines. With `k = 3`, there are `288`. 

For a fixed board size, smaller `k` means more possible winning windows. Larger `k` means fewer windows, because each candidate line needs more uninterrupted space.

This is why my 4x4, 3-in-a-row version felt so different to the children. More winning windows means more local threats. A move can matter in several directions at once. Forks become easier to create. Defence becomes less obvious because blocking one threat may leave another one open. 

The top row of a 4x4 board already shows the problem:

```text
. X X .
```

With `k = 4`, this row is not an immediate emergency. X still needs both empty ends to complete the full row. With `k = 3`, either end is a winning move:

```text
X X X .   or   . X X X
```

Diagonals create the same effect. This position is harmless in the 4-in-a-row version, but one move from over in the 3-in-a-row version:

```text
X . . .
. X . .
. . . .
. . . .
```

If X plays the third diagonal square, the game ends:

```text
X . . .
. X . .
. . X .
. . . .
```

That is why children who were comfortable with normal TicTacToe suddenly began missing threats. The board did not look crowded, but the number of short lines hiding inside it had grown.

This does not prove that X always wins as the board grows. That would be too strong. What it does show is that `k` is not a minor rule parameter. It changes the shape of the search problem.

## Where the Problem Suddenly Changes: Phase Transition

Algorithm designers care about these turning points.

When `k` is close to `n`, wins are long and fragile. A single blocking mark can spoil a whole line. Draws become more plausible because players have enough room to interrupt one another.

When `k` is small compared with `n`, winning windows multiply. The board becomes full of short local races. It starts to resemble games like Gomoku, where threats and counter-threats stack quickly.

Somewhere between those extremes, the game changes character. It may shift from draw-heavy to win-heavy. It may shift from easy to search exhaustively to completely impractical. That kind of sharp behavioural change is often called a phase transition.

I point at a useful algorithmic question: as `n` grows, how should `k` grow if we want the game to stay balanced and analyzable? 

A rough way to see the pressure is to look at the number of winning windows. If `k = n`, the count grows only linearly with `n`. If `k` stays fixed, the count grows quadratically with the board. Those are very different worlds.

That is the kind of question that turns a children's game into research fuel.

## Why This Matters for AI

Classical AI did not begin with neural networks. A lot of it began with search.

Take a game state. Generate legal actions. Apply one action to produce a new state. Check whether the new state is terminal. Repeat until the tree ends, then work backward to choose the best move.

That is the heart of tree search (think Breadth-First Search, BFS, and Depth-First Search, DFS). Making the search tree aware of two opposing players gives us MiniMax. Add alpha-beta pruning and you can ignore branches that cannot affect the final decision. Add better move ordering and the pruning improves. Add a transposition table and you avoid re-solving positions you have already seen. These are not decorative tricks; they are the difference between a solver that finishes and a solver that drowns.

Exact search has a hard ceiling. If the tree is too large, the correctness of BFS or DFS does not help you much. An optimal algorithm that cannot return before dinner is not useful during a game.

That is where Monte Carlo Tree Search enters the story. Instead of expanding every possible future, it samples futures and spends more effort where the search looks promising. AlphaZero-style systems go one step further: they use neural networks to guide search toward moves worth considering.

The machine does not become magical. It becomes selective. 

And that is the second lesson. Real-world AI often hits the same wall that bigger TicTacToe hits: too many possible states, too many action sequences, too many futures to enumerate exactly. Planning, routing, scheduling, protein folding, game playing, program synthesis; once the branching factor grows, brute force collapses.

TicTacToe just lets us watch the collapse happen on a board small enough to draw by hand.

## What I Took Away

I started with a game the children already loved. I made the board bigger. Then I changed one number.

That was enough to move from playground strategy to deep recursion, combinatorial explosion, phase-transition intuition, and explaining the limits of exact AI search.

The funny part is that the children understood the important bit before the math appeared. They could feel that the game had changed. More space did not simply mean more freedom. A smaller `k` did not simply mean an easier win. The rules interacted, and the board became harder to reason about.

That is algorithm design in miniature.

In the next post, I will take this broken version of TicTacToe and treat it the way classical AI would: as a search problem. We will start with MiniMax, sharpen it with alpha-beta pruning, and see exactly where exact search begins to run out of breath.
