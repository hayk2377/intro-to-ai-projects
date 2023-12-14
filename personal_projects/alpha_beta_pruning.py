from copy import deepcopy
import random
def get_moves(board):
    moves=[]
    for i in range(3):
        for j in range(3):
            if board[j][i]=='':
                moves.append((i,j))
    return moves
def make_move(board, move,player):
    b=deepcopy(board)
    x, y = move
    b[y][x] = player
    return b
def is_game_over(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != '':
            return (1 if board[row][0] == 'X' else -1)
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != '':
            return (1 if board[0][col] == 'X' else -1)
    if board[0][0] == board[1][1] == board[2][2] != '':
        return (1 if board[1][1] == 'X' else -1)
    if board[0][2] == board[1][1] == board[2][0] != '':
        return (1 if board[1][1] == 'X' else -1)
    if is_tie(board):return 0
def is_tie(board):
    for row in range(3):
        for col in range(3):
            if board[row][col] == '':
                return False
    return True
def evaluate(board,player):
    row_score=[0,0,0]
    col_score=[0,0,0]
    diag_score=0
    anti_diag=0
    for i in range(3):
        for j in range(3):
            if board[i][j]=='X':
                if i+j==2:
                    anti_diag+=1
                if i==j:
                    diag_score+=1
                row_score[i]+=1
                col_score[j]+=1
            if board[i][j]=='O':
                if i+j==2:
                    anti_diag-=1
                if i==j:
                    diag_score-=1
                row_score[i]-=1
                col_score[j]-=1
    x_count=0
    o_count=0
    for row in row_score:
        if row==2:
            x_count+=1
        if row==-2:
            o_count+=1
    for col in col_score:
        if col==2:
            x_count+=1
        if col==-2:
            o_count+=1
    if diag_score==2:
        x_count+=1
    if diag_score==-2:
        o_count+=1
    if anti_diag==2:
        x_count+=1
    if anti_diag==-2:
        o_count+=1
    if x_count>=1 and player=='X':return 1
    if o_count>=1 and player=='O':return -1
    if abs(x_count-o_count)==2:return (x_count-o_count)/2
    return 0
def alpha_beta_pruning(board=[["", "", ""], ["", "", ""], ["", "", ""]], maximizing_player=True, depth=4, alpha=-100, beta=100):
    if is_game_over(board):
        return is_game_over(board)
    if maximizing_player:
        player='X'
    else:
        player='O'
    if depth == 0:
        return evaluate(board,player)
    if maximizing_player:
        best_score = -200
        for move in get_moves(board):
            new_board = make_move(board, move,"X")
            score = alpha_beta_pruning(new_board,False, depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score

    # If the current player is minimizing.
    else:
        best_score = 200
        for move in get_moves(board):
            new_board = make_move(board, move,"O")
            score = alpha_beta_pruning(new_board,True, depth - 1, alpha, beta)
            if score < best_score:
                best_score = score
                beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score

print(alpha_beta_pruning([["X", "", ""], ["", "", ""], ["", "", "O"]],False))
        