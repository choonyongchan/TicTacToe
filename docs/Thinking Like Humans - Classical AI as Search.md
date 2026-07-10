* Searching For Answers Like Humans - Classical AI as SearchFr

* In the previous post, we discovered how TicTacToe, a simple game understood, enjoyed, and strategised by young preschoolers, can become complex once we increases the board size beyond 3x3 (increase beyond 2 dimensions, or increase beyond 2 players).

* We also submit that mathmatical analysis of TicTacToe is more complex than expected, since TicTacToe's rules allow for 'early wins'. Since there exists 'early stopping', there is no simple, repeatable pattern across the game as the board size increases, which hampers our analysis. Without a closed mathematical form, and only with recursive, exponential-growing mathematical calcuation, even a simple question like calculating the number of unique states in a nxn TicTacToe game requires an exponentially expensive search.

* Faced with this daunting task of a nxn TicTacToe board where a player wins with k consecutive symbols in a row, how can we begin to use AI to solve a game of TicTacToe? 

* "AI as a search", classified under Classical AI, converts our fundamental knowledge of Data Structure and Algorithms to convert AI problems into a search problem, where we can use search algorithms (e.g. Breadth-First Search, Depth-First Search, Iterative Deepening Search) to find a path from our root node (start state) to the goal node (intended end state).

** Classical Search

* Hold on, how then we can represent real-world problems into its mathematical equivalent? Well, search problems have to fit into these general structure: State, Transition, Action, Rewards, Terminal/Goal (I remember it as START).

** State, s, is the mathematical representation of the problem environment. What does it mean? For example, a 3x3 TicTacToe board can be represented by a 2D Matrix or a List of 3 rows, each with 3 elements. <insert a generated image here of how a 3x3 TicTacToe board is equivalent to a 2D matrix or a list of 3 rows with 3 elements>. State representation is not unique; there are many possible ways to represent a state. 
For real world problem, it is impossible and infeasible to represent every microscopic data in the environment. Imagine accounting for weather, relative humidity, and player's hydration level when analysing a game of TicTacToe: Not pertinent. So the state representation is merely a subset of pertinent elements of the problem environment. Moreover, the state representation can only contain observable elements of the problem environment. Training a poker model knowing every opponent's hand is not at all helpful: in practice, we do not have access to these information and the model will be rendered unhelpful. In AI terms, being able to observe the entire environment is called "Fully Observable" like in TicTacToe and Chess, while being able to observe only a portion of the environment is called "Partially Observable" like in Autonomous Driving and most card games. Simply put, the state is what the agent (i.e. player) can observe/see. Therefore, it is sometimes good to include more relevant details in the state. However, we should make sure the state does not include irrelevant details nor miss out on important details. The old adage says that "knowing is half the battle", so knowing the optimal way to represent the problem environment in a mathematical state is an art, and an active area of research. (see State Design) 
In AI terms, the set of every possible state is a constant known as the "State Space", and it can be exponentially magnificant.

** Transition, T, describes how a state change to another state, once the state is mathematically defined (i.e. s -T-> s'). For example, placing an 'X' on the top-left empty spot involves a transition from a 2D matrix where matrix[0][0] = null to the exact same 2D metrix where matrix[0][0] = "X". Understanding the transition helps to verify the correctness of implementation. In our definition of a transition, we can verify the correctness by checking if the matrix's shape is always (3,3) and the difference of number of nulls in the old matrix and number of nulls in the new matrix is always 1 (i.e. numNull(m) - numNull(m') = 1)

** Action is an abstract list of what an agent (i.e player) can do at a state, s' = A(s). For example, in PacMan, users can traverse up, down, left, right. Each action, A, to a state, s, corresponds to a valid transition to a new state, s'. In TicTacToe, each empty grid is an available action that an agent can do. The set of every possible action is known as the Action Space. However, depending on the state, not all actions in the action space is possible. 

** Goal is an abstract condition for search termination, which has to be represented as state(s). Sometimes, we are only interested in a particularly simple condition, which encompasses many states. In that case, all states that fulfil that condition are the goal.
For example, in TicTacToe, it is difficult to map all the goal state because there are too many. Instead, we define a goal condition. When the goal condition is reached, search terminates when any straight line of 3 same symbols is found. Mathematically, I will check if a goal is reached if a line have elements with the same symbol for all rows, columns and diagonals. <Here is the pseudocode: >

** Reward, R(s,a), is optional. Some people just want to find any path to any goal, while other people wants to find the "optimal" path to any goal or the "optimal" goal. The definition would be different, for example "win with the least number of moves". Reward is implemented to incentivise mathematical models towards "optimal" actions and/or penalise "suboptimal" actions. Rewards can be assigned after the goal is reached (i.e terminal reward), or after each move (i.e. intermediate rewards). Rewards are assigned to a state, (an end state), and an action(s), R(s,a) or R(s,a,s').
Similar to state representation, reward design is an art. In TicTacToe, we know that an obvious winning move should be rewarded vis-a-vis a meaningless move, because we want the model to swiftly deal the final blow. However, given two purposeful moves from an intermediate state, how can we determine which move is "better" than the other move? Surely, the specific state must be taken into account here. For example, a move in the center is worth more at the start, than towards the end, when the winning move is at the board edge.
Reward design is a fragile art. Carelessly incentivising the model will train the model to be an expert in careless actions. Here is a thought experiment: Is it always true that a move in the centre is worth more than the move is at the board edge? Perhaps at the start to dominate the board, but not when a winning line can be formed at the board edge. Therefore, in this case, an intermediate reward must be carefully implemented with rigorous analysis or abandoned in favour of a terminal reward which rewards the action sequence that led to the win.

* Search Tree

<insert an image of a boy searching for a tree in a garden>.

In TicTacToe, consider the empty board (i.e. start state) as the root node of a search tree.
Each action is a tree branch, connecting the empty board, to a new board with 1 symbol on it. Therefore, the branching factor of the tree, b, is the number of valid actions. After 1 move, there are about b new possible boards.
Moving forward, each action connects the new boards with 1 symbol on it, creating new boards with 2 symbols on it for each new board with 1 symbol on it. After 2 moves, there are about b(b-1) new boards, since the new board has one less grid to act on. Notice that the search space is gorwing exponential..
The tree stops growing when the goal is reached. Intuitively, the least number of moves to reach a goal is 5 moves (or 2k-1 moves, if k is needed to win), and the most number of moves to reach a goal is 9 moves (or n^2, where all grids are exhausted).

In big-O notation, let the branching factor be b and the max depth of the tree be d, then the number of possible states grow in the order of O(b^d). For TicTacToe, the number of possible states grow in the order of O(max number of actions ^ n^2) = O((n^2)^(n^2)).

** AI as a search

* "You said find a path from root to goal in a tree?" A up-and-coming Computer Scientist would quickly attempt BFS or DFS to solve this game tree. Good intuition! With patience (lots of patience, O(b^d) amount of patience), a terminal board and the path towards the goal will be found.

* However, the player are playing against an enemy. How should we account this in our BFS/DFS algorithm?

1. MiniMax
<insert a picture of MiniMax found online>

MiniMax <link to geeksforgeeks> is a modified DFS search algorithm that works for Zero-Sum, Turn-Based games. The implementation involves a maximiser (player 1) and a minimiser (player 2), who seeks to maximise and minimise the reward respectively as every turn. For example, if player 1 reaches a goal, that terminal node is worth +1 reward. Otherwise, if player 2 reaches a goal, that terminal node is worth -1 reward. Propagating the terminal rewards from the terminal nodes back up to the root node, the maximiser assumes its minimiser opponent acts rationally and will pick the minimal reward at the next turn. Therefore, for this turn, the maximiser will pick the action that maximises the reward the minimiser gives him. In simple English, the maximiser will pick the action which disadvantage the minimiser the most even if the minimiser plays perfectly. Likewise for the minimiser against the maximiser. The MiniMax algorithm conservatively assumes that each player has perfect foresight, and makes perfectly optimal moves at each move. This could mean that the MiniMax algorithm could miss opportunistic 'early wins' if the opponent were to 'fumble' or be inexperienced.

The benefit is that each intermediate state has a "reward" attached based on the future actions and states, and can be evaluated against other intermediate neighbouring states to determine the optimal actions. Moreover, as an exact search algorithm, MiniMax is guaranteed to give the optimal action at each state (that is, if it is even able to calculate that far deep).

The evaluation is painful, as every state must be reached to account for all possible states and their rewards. Being a DFS-based algorithm, if the search stops prematurely, there is no guarantee that the current best action is anywhere near the actual best action, since other second-move actions have never been explored at all. For large search state, MiniMax is fantastic only in theory.

Fortunately, there are optimisations specific to the idea of maximisation-minimisation that we can implement. 

2. MiniMax with Alpha-Beta Pruning.
"Never step where your enemy's best move is obvious".

A smarter player would look one move deeper to analyse his opponent's thought process. This works because MiniMax defines the opponent's thought process mathematically. If an action gives the minimiser their strongest move, then the maximiser has no business exploring further. Thus, when the maximiser finds that the minimiser has a very strong move, the maximiser need not search other adjacent branches, because the minimiser would pick this strong move or other even stronger moves and the maximiser would definitely be worse off venturing into this action.

This logic is implemented using an upper and lower bound, alpha and beta, meant for the maximiser and minimiser respectively. 

For Maximizing Player, it updates the best value found so far and raises alpha; pruning occurs when beta <= alpha.
For Minimizing Player, it updates the minimum value and lowers beta; similarly prunes unnecessary branches. (Remember this intuitive, we'll use this extensively to understand null-window search)

Alpha-Beta pruning allows MiniMax to return the same results while ignoring some (hopefully many) irrelevant states and actions, thus improving search time complexity. 

However, alpha-beta pruning heavily relies on the search order, or "move ordering". Simply explained, if the maximiser manage to explore his strongest move first, using his strongest move as the reference (via updated alpha-beta), it would not take long for the maximiser to discover that other weaker move would give less optimal rewards, and many sub-branches would be ignored.

According to research, the search time complexity would be reduced from O(b^d) to O(b^d/2) at best. This means that for the same time, the MiniMax with Alpha-Beta Pruning can search twice the depth, or that a search that takes 100 hours will now take sqrt(100) = 10 hours, an exponential reduction.

Moving forward, let's assume that Alpha-Beta Pruning is implemented, since it is an optimisation with negligible memory complexity added.

3. NegaMax 

A sharp observer figured that for zero-sum, two-player games, the maximiser and minimiser are acting identically, but in opposite directions! So instead of having a maximiser, a minimiser, an alpha, and a beta, we'll only consider a maximiser and an alpha. Whenever there is a switch of player, the compiled reward score will be negated. This intuitively means that player will try to maximise their own reward according to their own terms! 

Fundamentally, the algorithm has never changed, except that there are fewer variables to track, and implementation becomes simpler.

(As software engineers, we know when the codebase becomes too complicated, the KISS principle is just like your first kiss, bringing warmth and comfort to our hearts.)

4. NegaMax with Alpha-Beta Pruning and Heuristics

As seen from the improvement from MiniMax to MiniMax with A-B pruning and to NegaMax, we can further optimise our algorithm by incorporating knowledge about our environment. I mean, it is intuitive that a person who have learnt some tips and tricks would play the game better than a person who learnt the game for the first time.

Similarly, there are two types of heuristics: domain-agnostic heuristics and domain-aware heuristics. Domain-agnostic heuristics are tied to the algorithm and are independent of the environment itself. For example, I can use A-B pruning whenever I implement MiniMax to a search problem. I do not need to care what the search problem is. However, domain-agnostic heuristics can only bring us so far. 

Domain-aware heuristics incorporate both the algorithm and the environment to optimise the algorithm especially in efficiency. For example, in a 3x3 TicTacToe game, whenever player X sees an almost winning line on his turn (e.g. X_X), player X should most definitely place his piece in the middle to win, instead of searching for more actions. This particular "winning-move" heuristic is a domain-aware heuristic, and is only defined for this 3x3 TicTacToe scenario. A 4x4 TicTacToe with 4-in-a-line cannot use the same heuristic.

What domain-aware heuristics are there for TicTacToe? One possible heuristic is Scaled Rewards.

** Scaled Rewards: The heuristic that changed the question
As mentioned, we can score a win as early as 2k-1 moves into the game, where k is the number of consecutive symbols we need to form a winning line. We could also score a win as late as n^2 moves into the game, where we exhaust all grids in the board. 
The intuition is that, if we can incentivise actions which leads to an early win, then we can prune away other actions which takes a deeper depth regardless whether it takes a win or loss. 
Why search deeper when we can attack an action that wins earlier?
<insert an image about the first goal setting the depth limit of search>

Sounds great!... in theory. A seasoned algorithms engineer would have noticed that the question has been changed with this heuristics, leading to an unintended outcome.

Before scaled rewards, the "reward space" is only {-1, 0, +1}. Once an action produces a +1 reward for the maximiser, the maximiser has obtained the maximal possible reward for the game, and need not search any more actions any longer. Why? By MiniMax's definition, even when the minimiser opponent tries its best, the worst possible reward for this particular action is still +1, the absolute best reward for the maximiser. Since that action will guarantee a win against even an optimal player, the player need not search any further. 

By introducing scaled reward, even though I am shrinking the search space, I am forcing MiniMax to find the least-move win instead of any win. Even if I have found a guaranteed win with this action by depth x, I must still continue the search because I could have a guaranteed win with another action in a shorter depth. This fundamentally changes the question, and incurs a search penalty. Do I need the least-move win? No, my goal is to make finding any goal faster and more efficient.

4. NegaScout

Recall that in Alpha-Beta Pruning, pruning only occurs when beta <= alpha. So, why not increase the chances that beta <= alpha, so that more pruning can happen! How can we do so? 

Assuming the first action we search is the best action, then the updated alpha, which is the current best score, is presumably the actual best score we can get. Then, we just need to prove that other actions are worse or equally as good as this first action by using alpha as a lower bound. Any actions which give a smaller value can be quickly pruned away. However, if there are any action that can give any result better than my presumed best action, then my assumption is disproven, and I'll have to do the standard Negamax search with Alpha-Beta pruning (no additional pruning). My upper bound must be infinitesimilarlly smaller than alpha in order to let any better results exceed the my upper bound. Practically, my upper bound is "alpha + \delta", where \delta is the smallest increment between any two MiniMax values. Usually, \delta is 1, but it can be smaller like in our "scaled rewards" methodology. 
In summary, we have changed the standard Alpha-Beta pruning bound from [alpha, beta] to [alpha, alpha + \delta], in hopes that at every depth, more nodes are pruned. This bound is called Null Window.

NegaScout is a variant of null-window search, which uses this null-window strategy to improve the pruning ability over Alpha-Beta pruning. However, the effectiveness of NegaScout is heavily dependent on Move Ordering or Principal Variation. Simply explained, if the first move is objectively the best move, then the updated alpha is the maximum reward the player can receive. Pruning efficiency is maximised, because it would not take long for NegaScout to find out that other actions cannot get a reward close to the current alpha. However, if the first move is objectively a bad move, then the updated alpha is not the best reward the player can receive. Pruning efficiency is minimised, because when NegaScout find another action that offers a higher reward than the first action, a full NegaMax search must be done on that action to obtain the 'better' alpha.
NegaScout is another example of how knowing the state well enables domain-specific optimisations. Loosening of the said assumptions about the state will degrade performance but fortunately not by much.

5. MTD(f)

MTD(f) is seen as a successor of NegaScout. Instead of using a mix of Null-Window Searches and full NegaMax searches, MTD(f) exclusively uses Null-Window Searches to maximise pruning. This is possible with the clever use of binary search. In essence, MTD(f) "guesses" the value of alpha, and iterative evaluation of MTD(f) converges the value of alpha to its true value. 

The algorithm works by calling NegaMax a number of times with a null search window. The search works by zooming in on the minimax value. Each AlphaBeta call returns a bound on the minimax value. The bounds are stored in upperbound and lowerbound, forming an interval around the true minimax value for that search depth. When both the upper and the lower bound collide, the minimax value is found.

MTD(f) gets its efficiency from doing only zero-window alpha-beta searches, and using a "good" bound to do those zero-window searches. Conventionally AlphaBeta is called with a wide search window, making sure that the return value lies between the value of alpha and beta. In MTD(f) a window of zero size is used, so that on each call AlphaBeta will either fail high or fail low, returning a lower bound or an upper bound on the minimax value, respectively. Zero window calls cause more cutoffs, but return less information, only a bound on the minimax value. To nevertheless find it, MTD(f) has to call NegaMax a number of times, converging towards it. The overhead of re-exploring parts of the search tree in repeated calls to NegaMax disappears when using a version of NegaMax that stores and retrieves the nodes it sees in memory.

In order to work, MTD(f) needs a "first guess" as to where the minimax value will turn out to be. The better than first guess is, the more efficient the algorithm will be, on average, since the better it is, the less passes the repeat-until loop will have to do to converge on the minimax value. If you feed MTD(f) the minimax value to start with, it will only do two passes, the bare minimum: one to find an upper bound of value x, and one to find a lower bound of the same value.

Compared to NegaScout, MTD(f) simplies the code significantly and pushes the complexity to the root driver loop. Since each call to NegaMax are somewhat independent, MTD(f) can be parallelised during evaluation.

As mentioned, an assumption is that MTD(f) requires a well-implemented Transposition Table to prevent repeated evaluation on repeated calls to NegaMax. This invokes a memory tradeoff. Also, MTD(f) is sensitive to a poorly-implemented Transposition Table, and will greatly suffer from re-searches. 

6. Best Node Search

Rather than trying to find out the exact numerical value of a node, why not focus on the goal of figuring out which move is strictly the best?

Best Node Search uses successive approximation to distinguish between the best move and the second-best move. It makes a guess at a value, and then counts how many moves are worse or better than the guess. By adjusting the guessed score, it progressively eliminates sub-optimal branches ("disprove the rest") or narrows the value of the optimal branch ("prove the best") until only the best possible move remains.

Here is an example
Is uses successive approximation to find a numerical evaluation between the best move and the second best move.
For example
--
Guess .23 ?
11 moves are worse than .23
9 moves are better than .23
--
Guess .43 ?
20 moves are worse than .43
0 moves are better than .43
--
Guess .33 ?
18 moves are worse than .33
2 moves are better than .33
--
Guess .38 ?
19 moves are worse than .38
1 move is better than .38
--
Aha that's the best move.
We don't know the exact evaluation, but it's less than .43 and greater than .3

Best Node Search is more efficient than MTD(f), especially for larger search spaces because irrelevant moves to search for the "best value" is skipped when the algorithm knows that no other moves give a result that is better than the "best value". This can be seen as "early stopping". 
Similar to MTD(f), Best Node Search starts with an initial "best value", in which performance hinges on. A good initial "best value" reduces the number of iteration needs to isolate the best action, improving efficiency. 
Empirical research shows that Best Node Search performs better than all other search algorithms. 
Note: Iterative Deepening 

Iterative Deepening search is a DFS search that increases in depth at each iteration. This provides the performance of BFS (of quickly finding wins at a limited depth) while having the linear memory complexity of DFS. The iterative nature of Iterative Deepening Search theoretically does not significantly increase the search complexity, since the search tree is already exponential in nature. The theory goes that the number of nodes in the deepest depth typically exceeds the sum of the number of nodes in all previous depths combined.

It is worth noting that BFS can be seen as a Iterative Deepening search where the frontier is stored in memory to be further explored when needed from that node. This intuition can be  used in NegaScout to get the current best action, MTD(f) and Best Node Search to get the current best initial value.

In NegaScout, move ordering is crucial to its efficiency. It is difficult to determine which move ordering is the most optimal, especially at each depth. With iterative deepening (and a good design of intermediate rewards), the best move from depth D can be determined. Since the game of TicTacToe requires a contiguous line to be form, the subsequent moves would most probably be moves that encircle the previous best moves. This is just one way of using information from previous depth to squeeze some information about the most optimal move order.

As mentioned, MTD(f) works best with a good "first guess". Using iterative deepening, we can store the best value retrieved at the previous depth as the "first guess" for the next iteration. Though imperfect, it is much better than throwing a no-knowledge "first guess". Of course, iterative deepening introduces another problem: the need to determine the value of intermediate nodes. This shifts the problem from retrieving the best possible "first guess" to determine the value of intermediate nodes. As mentioned, designing the reward for intermediate nodes is an art, but fortunately, the reward need not be accurate but just a guess. It is also fortunate that research is active in this artistic area of reward design. 
Similar to MTD(f), Best Node Search also uses  iterative deepening to generate a rationally "good" initial value based on the previous depth's value for seemingly negligible performance impact. 

What I learnt (in general):
1. Today's technology is built upon many iterative generations of innovations and ingenuity. Some algorithms might seem trivial to a junior undergraduate student, but they have taken many years to discover. Without simple stepping stones and discovery, we might not have discover such new algorithms to make problems easier to solve. 
With the new age of quantum technology and intractable problems about Explainable AI and the pushing the frontiers of AI, I have learnt not to trivialise the small wins in the field, and to not let the daunting nature of the topic discourage me from trying and exploring.
2.  Loosening assumptions and using heuristics is not too bad after all.
There are many fields where accuracy is paramount: medical and logistics, to name a few. However, there are many other fields where finding a quick solution is much more important: autonomous driving, real-time operating system.
In many other areas of life, heuristics is important too. Leaders are always bombarded with many decisions to make daily, and do not always have the luxury of time to make decisions. With simple "rule of thumb", heuristics enable decisions to be made fast (resource efficiency), simple (complexity reduction), transparent and efficient. However, it is up to us to find which heuristics most accurate reflect the real world. This could mean defining heuristics more strictly (e.g. dominant heuristics and admissible heuristics in pathfinding algorithms). 
3. Domain knowledge is important
The most effective heuristics are domain-aware heuristics, the heuristics that knows the problem environment deeply and accounts for the environment within its heuristic calculation. 
The world is complex, and very much so now. It would be a fallacy to presume that the understanding of the world can be effectively reduce to heuristics: that would make us skip important nuances and make sweeping generalisations. However, it helps to be "domain-aware", being increasingly aware of my surroundings. I am encouraged to read widely, converse with more people around me. I believe that these connections and knowledge will make me a holistic individual who may not be a master, but a jack of all trades: a jack who can not only take care of children in a volunteering center, but also thinking deeply about algorithms <see my previous post here https://choonyongchan.github.io/thoughtsofaservant/algorithms/i-ruined-tictactoe-for-my-children-and-for-math/>.


Food for thought
1. Is finding a closed form for this recursive relation truly intractable? Or is it the closed form merely convoluted?
2. Besides "instant win" heuristics and intermediate reward, what are some other provably-effective low-cost heuristics for TicTacToe? Killer move heuristics (i.e. forward-thinking moves to ensnare your opponents?) 
3. Do these observations and heuristics hold true for p-player nxn tictactoe, or n-dimensional tictactoe board?
