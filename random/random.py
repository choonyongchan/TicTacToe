import numpy as np

def check_win(board, k):
    n = len(board)
    for i in range(n):
        for j in range(n):
            if board[i][j] != 0:
                player = board[i][j]
                # check row
                if j + k <= n and all(board[i][j + p] == player for p in range(k)):
                    return player
                # check col
                if i + k <= n and all(board[i + p][j] == player for p in range(k)):
                    return player
                # check diag
                if i + k <= n and j + k <= n and all(board[i + p][j + p] == player for p in range(k)):
                    return player
                # check anti-diag
                if i + k <= n and j - k + 1 >= 0 and all(board[i + p][j - p] == player for p in range(k)):
                    return player
    return 0

def get_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves

def dfs(board, player, k):
    win = check_win(board, k)
    if win == 1:
        return (1.0, 0.0, 0.0)
    elif win == -1:
        return (0.0, 1.0, 0.0)
    moves = get_moves(board)
    if not moves:
        return (0.0, 0.0, 1.0)
    total_X = 0.0
    total_O = 0.0
    total_draw = 0.0
    for move in moves:
        new_board = [row[:] for row in board]
        new_board[move[0]][move[1]] = player
        px, po, pd = dfs(new_board, -player, k)
        total_X += px
        total_O += po
        total_draw += pd
    num_moves = len(moves)
    return (total_X / num_moves, total_O / num_moves, total_draw / num_moves)

# Example usage
n = 4
k = 4
board = [[0] * n for _ in range(n)]
prob_x_win, prob_o_win, prob_draw = dfs(board, 1, k)
print(f"Probability of X winning: {prob_x_win}")
print(f"Probability of O winning: {prob_o_win}")
print(f"Probability of draw: {prob_draw}")