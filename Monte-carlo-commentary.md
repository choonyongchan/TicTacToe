[10/7/2026 1:03 pm] choonyy: I think this is an excellent direction because n×n Tic-Tac-Toe sits in a sweet spot: it's simple enough that every algorithm can be implemented from scratch, yet large enough that the progression from classical search to modern AI becomes very natural. Students can literally see why each generation of algorithms was invented.
Rather than presenting algorithms chronologically, I'd present them as successive solutions to the limitations of the previous approach.
Era
Core idea
Why it exists
Exhaustive Search
Search everything
Works only for tiny games
Pruned Search
Search less
Larger search trees
Heuristic Search
Estimate leaf quality
Can't reach terminal states
Statistical Search
Sample instead of enumerate
Branching factor too large
Learning-based Search
Learn the heuristic
Hand-crafted heuristics don't scale
Self-play Learning
Learn from experience
No human knowledge required
Model-based Planning
Learn the game itself
Simulator unavailable
General Reasoning
Learn to plan
Beyond games
This tells a coherent story instead of presenting disconnected algorithms.
Part I — Classical Search (the foundation)
You've already explored these, but I'd include them briefly because they motivate everything that follows.
1. Minimax
The first complete decision algorithm.
Students learn:
adversarial search
game trees
utility functions
optimal play
2. Alpha-Beta Pruning
First lesson:
You don't have to search everything.
Introduce:
pruning
move ordering
exponential improvements
3. Advanced Alpha-Beta Variants
These show that engineering matters.
Examples:
Negascout (Principal Variation Search)
MTD(f)
Best Node Search
SSS*
Transposition Tables
Iterative Deepening
These remain surprisingly competitive today.
Part II — Heuristic Search
Here comes the first big conceptual shift.
Instead of
Search until game over
students learn
Search a little

↓

Estimate remaining game
Topics:
evaluation functions
feature engineering
pattern recognition
Example features:
open rows
forks
center control
mobility
threats
Students quickly discover that designing heuristics is difficult.
This motivates machine learning.
Part III — Monte Carlo Methods
This is where AI starts to feel modern.
1. Pure Monte Carlo
Random playouts.
Move

↓

1000 random games

↓

Average outcome
Students immediately see
noisy estimates
statistical convergence
exploration
2. UCT
One of the most beautiful algorithms in AI.
Instead of
Always choose best move
UCT asks
Should I exploit

or

Should I explore?
This introduces
multi-armed bandits
Upper Confidence Bounds (UCB)
exploration vs exploitation
One equation explains much of modern reinforcement learning.
3. Monte Carlo Tree Search
This is the natural culmination.
The four phases are elegant:
Selection

↓

Expansion

↓

Simulation

↓

Backpropagation
Students can visualize each phase easily.
Part IV — Learning
This is where AlphaGo begins.
Instead of
Human writes evaluation function
we have
Neural network learns evaluation.
Policy Networks
Predict
Which move looks promising?
instead of
Which move is best?
Value Networks
Predict
Probability of winning
instead of searching to the end.
MCTS + Neural Networks
Now MCTS becomes
Search

guided by

learning.
This is the AlphaGo insight.
Part V — Self-play
A wonderful concept for students.
Instead of
Teacher

↓

Student
we have
Student

↓

plays itself

↓

becomes teacher
Concepts:
bootstrapping
curriculum emergence
policy improvement
policy iteration
Part VI — AlphaZero
I'd spend considerable time here because it's one of the cleanest algorithms in AI.
Pipeline:
Current network

↓

Self-play

↓

Generate games

↓

Train network

↓

Improved network

↓

Repeat
Students often realize this resembles gradient descent combined with search.
Key idea:
Search improves learning.
Learning improves search.
This positive feedback loop is the heart of AlphaZero.
Part VII — MuZero
This is where modern AI becomes truly exciting.
Instead of assuming
Game rules known
MuZero learns
Representation

↓

Dynamics

↓

Prediction
[10/7/2026 1:03 pm] choonyy: It effectively learns an internal simulator.
This naturally leads into discussions of world models.
Part VIII — Frontier Research
Here I'd move beyond board games.
EfficientZero
Improves MuZero's sample efficiency.
Great topic because it addresses a practical weakness.
Gumbel MuZero
A fascinating improvement that replaces parts of MCTS with Gumbel sampling.
Students learn that even AlphaZero's search procedure is still being refined.
Dreamer
Planning entirely inside learned latent space.
Excellent bridge toward robotics.
World Models
Teach students the idea:
Think

before

acting.
The AI imagines futures before making decisions.
Transformers for Decision Making
Interesting research directions:
Decision Transformer
Trajectory Transformer
Planning becomes sequence prediction.
Students already familiar with language models appreciate this perspective.
Test-Time Search
Very current research asks:
Can a model perform more computation only when needed?
Examples include:
Tree-of-Thought
Best-of-N
inference-time planning
search over reasoning traces
Although developed for language models, these ideas are closely related to classical search.
Part IX — Research Questions for n×n Tic-Tac-Toe
This is where your project could become genuinely interesting.
Instead of merely implementing algorithms, ask questions such as:
Scaling
At what board size does:
minimax fail?
alpha-beta fail?
MCTS become superior?
AlphaZero outperform MCTS?
Students love empirical crossover points.
Branching Factor
How does performance change as
3×3

↓

5×5

↓

7×7

↓

15×15
The tree explodes dramatically.
Training Cost
Compare:
search time
training time
inference time
memory
playing strength
Sample Efficiency
Which algorithm needs fewer games to become strong?
This is an active research topic.
A chapter sequence I'd recommend
If I were writing this as a textbook or tutorial series, I'd organize it as follows:
The Game Tree — Minimax and adversarial search.
Searching Smarter — Alpha-beta pruning, transposition tables, and advanced alpha-beta variants.
When Search Is Not Enough — Heuristic evaluation functions.
Learning by Simulation — Monte Carlo methods and MCTS.
Learning to Evaluate — Neural policy and value networks.
Learning Without Humans — Reinforcement learning and self-play.
AlphaZero — The search–learning feedback loop.
MuZero — Learning the game model itself.
Beyond Board Games — World models, latent planning, transformers for decision making, and inference-time reasoning.
Open Problems — Efficiency, scalability, and general intelligence.
A unifying message
The most memorable takeaway for students is that the field has been moving toward learning more and hard-coding less.
Minimax
│
├── Hand-crafted rules
│
├── Alpha-beta
│
├── Hand-crafted heuristics
│
├── Monte Carlo simulation
│
├── Learned evaluation (AlphaGo)
│
├── Self-play (AlphaZero)
│
├── Learned dynamics (MuZero)
│
└── Learned reasoning and planning (current research)
That progression gives learners a clear mental model: every major advance shifts another piece of intelligence—from human-engineered logic into learned representations. Using the same n×n Tic-Tac-Toe environment throughout lets readers isolate the effect of each algorithm, making the evolution of AI tangible rather than abstract.