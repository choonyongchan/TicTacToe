from src.agents.mtdf_id_agent import MTDfIDAgent
from src.core.state import State

agents = {p: MTDfIDAgent(max_depth=4**2) for p in ("X", "O")}
state = State(n=4, k=4)

while not state.is_terminal():
    player = state.current_player
    row, col = agents[player.name].act(state)
    state.apply(row, col)
    print(f"Player {player.name} plays ({row}, {col})")
    print(state.board.render())

winner = state.winner()
print("Winner:", winner.name if winner else "Draw")
