* Analysing Childhoodd Game - At Scale

Intro: Why I started this project.
1. My children in my volunteering centre love to play TicTacToe. It is a simple, quick game to bond with a friend.
2. One day, I challenged them to a 4x4 TicTacToe, and it got them to think a little harder. They realised that it is much easier to get into the stalemate, by blocking one of my moves.
3. My final challenge, is to get them to win me in a 4x4 TicTaeToe, where they only need 3-in-a-row to win. This seems to open up the world of possibilities to the children. Overzealous children were excited to compete with me. One thing I knew: a simple game of TTT is a great way to bond with children, and it's a SCALABLE game. 
4. As a Computer Science graduate who have been working with big systems, scalability has always floated up as the top of my concerns. In bid to refresh my memory on algorithms, I set out to explore the game of Scalable TicTacToe.
5. Coming from a 3x3 TicTacToe game, where we need 3-in-a-row to win, what can we learn from an nxn TicTacToe game, where we need k-in-a-row to win, where k<=n?
6. Subsequently, given a state, what algorithms are most suitable to optimally plan a game of nxn TicTacToe game, where we need k-in-a-row to win? And what does we learn about algorithms for such problems?

Paragraph 1: Characteristics of TicTacToe
1. Tic-Tac-Toe is a finite, deterministic, perfect-information game that has been completely solved through game-theoretic analysis. In standard 3×3 play, both players have full visibility of the board, there is no randomness in the rules, and the game lasts at most nine moves, giving it a fixed maximum depth. Despite its apparent simplicity, the game exhibits many of the core ideas studied in computer science and mathematics, including recursion, combinatorial explosion, symmetry reduction, dynamic programming, and adversarial search. The game tree branches rapidly as players explore different move sequences, yet many positions are equivalent under rotation and reflection, allowing the state space to be compressed significantly. Under optimal play, Tic-Tac-Toe always ends in a draw, making it a solved game in which neither player can force a win against perfect defense. Because of these properties, Tic-Tac-Toe is frequently used as a foundational model for studying artificial intelligence, search algorithms, probability, and game theory, while also serving as a useful lens for understanding how scalable combinatorial systems behave as their size and complexity increase.

Paragraph 2: Simplest analysis - Random Play
1. I remember my Kindergarden 1 child, sparring with me on TicTacToe randomly, and the child lost almost every game. I learnt from her tenacity because she tried and tried many times, always with a smile on her face. Of course, by pitting a Random agent with an Optimal agent, it is obvious who would win.
2. This gives me one question. Given 2 random agents, where X starts, what is the probability that player X wins?
3. Using TicTacToe(n=3,k=3) as an example, the probability is not trivial to find. it is hard to find a combinatorial patterns because for X to win, we must make sure that 1. O must not have a winning line anywhere, and 2. X must not have a winning line too. This intuitively implies that the solution is dependent on the previous move, and thus is recursive.
4. Using Mathematical Induction, let Px(moves) be the probability X will win given the current sequence of moves. Then
Px(moves) = \sum_{m_i=m_1}^{m_n} P(m_i)Px(moves + m_i), where at the terminal state, Px(moves) will be 1 is X wins, or 0 is the game ends in a draw or O wins.
5. Since the Random Agent randomly selects moves, P(m_i) = 1/n. Px(moves) = 1/n \sum_{m_i=m_1}^{m_n} Px(moves + m_i).
6. To find Px(moves), we will have to recursively enumerate through n subsequent moves and evaluate Px(moves + m_i). The calculation increases exponentially. 
7. Modelled as a tree, the branching factor is the number of empty squares, which decreases by 1 on each move. And the depth is variable, depending on how fast a win is achieved. The minimum depth is 2k-1, and the maximum depth is n^2. Since n>=k>=3, the maximum depth is always more than the minimum depth. 
8. To put numbers in perspective, a TicTacToe(n=3,k=3) game has __ states, and a TicTacToe(n=4,k=4) has ___ states. 
9. Since TicTacToe has "early stopping" when a win is detected, it is intractable to find a closed form for the calculation of the number of states as n and k scales.  

10. Under combinatorial explosion and a diverse terminal states, how can we analyse TicTacToe as n and k scale?
11. Well, without deeper analysis, let's explore these two cases.

12. When k=n, and n becomes larger, the probability of a win decreases exponentially towards 0. Remember that a win is only achieved with a winning horizontal line, winning vertical line, or a winning diagonal line. When k=n, there are n winning horizontal line, n winning vertical lines, and 2 winning diagonal line, a total of (2n+2) winning lines. However, as n becomes larger, the board size increases by n^2. Also, remember that to block the construction of a winning line, the opponent only needs one move in the winning line's path, and vice versa. This implies that, without a loss of generality, the probability of Player X winning tends to 0 when n becomes large, because the probability of a draw tends to 1.

13. When k remains constant at 3, and n becomes larger, the probability of a win increases linearly. Now, a win can be achieved on any consecutive k grids on a horizontal line, vertical line and diagonal line. To generalise, there are n(n-k-1) winning horizontal lines, n(n-k-1) winning vertical lines and (n-2) winning diagonal lines, a total of 2n(n-k-1)+n-2 = 2n^2 + (-2k+1)n + (-2n-2) winning lines. As n increases, the number of winning lines increases faster than the board size n^2. Therefore, especially when k is small relative to n, the number is winning lines is larger relative to the board size. Given that X is the first player, probability that Player X wins tends to 1. (AI to mathmatically explain why in depth)

14. Is there a "sweet spot" for k, such that when n becomes larger probability of Player X wins is 0.5? By the Mid Value Theorem, there must exist such a value. It is intuitively that k must grow according to n. Since when k = n, the probability of a win decreases exponentially towards 0, I conjecture that k = c log n, where c is an arbitrary constant. The existence of such boundary is an example of Phase Transition in algorithm analysis.

15. A phase transition in algorithms is a point where a problem suddenly changes from being “usually easy” to “usually hard” (or vice versa) as some parameter crosses a critical threshold.
The term comes from physics — like water abruptly freezing at 0°C — but in computation it describes sharp changes in algorithmic behavior.


16. Understanding Phase Analysis is important. They explain real-world system behavior
Many systems exhibit abrupt computational changes:
network congestion,
distributed consensus failures,
routing collapse,
cache thrashing,
combinatorial explosion.
Phase-transition analysis helps predict tipping points.

Paragraph 3: Prelude to algorithmic solvers

17. Now that we have a deeper appreciation of the complexity of TicTacToe at scale, let's explore algorithmic solvers of TicTacToe.

18. We will explore classical search (AI as search) which represents TicTacToe as a standard State-Transition-Action-Goal(-Constraints) format, and solve TicTacToe using tree-search algorithm. These algorithms are exact search and always gives the optimal move at the expoense of enumerating through all possible states.

19. Next, we will explore Monte Carlo Tree Search as a way to tackle exponential search spaces. We will discuss explore and exploit dichtomy.

20. Finally,we will explore AlphaZero-style Monto Carlo Tree Search, the same architecture that brought chess grandmasters and Go chmapions to their knees. While the use of Neural Networks as heuristics was innovative, we will firther explore Neural Network optimisations to reduce the heuristic cost of an AlphaZero-style solution.

21. Catch you in the next post!
