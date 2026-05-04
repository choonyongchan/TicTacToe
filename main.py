import time

from src.agents.mtdf_id_agent import MTDfIDAgent
from src.core.state import State

agents = {p: MTDfIDAgent(max_depth=5**2) for p in ("X", "O")}
state = State(n=3, k=3)

start = time.time()
while not state.is_terminal():
    player = state.current_player
    row, col = agents[player.name].act(state)
    state.apply(row, col)
    print(f"Player {player.name} plays ({row}, {col})")
    print(state.board.render())
elapsed = time.time() - start

winner = state.winner()
algo = type(agents["X"]).__name__
print(
    f"Winner: {winner.name if winner else 'Draw'} "
    f"| Algorithm: {algo} "
    f"| States visited: {state.state_count} "
    f"| Time: {elapsed:.2f}s"
)
