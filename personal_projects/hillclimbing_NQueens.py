import random
from copy import deepcopy
def set_board(size):
    board=[]
    for i in range(size):
        board.append(i)
    random.shuffle(board)
    return board
def fitness_function(board):
    score=0
    n=len(board)
    for col in range(n):
        queen=board[col]
        for other_col in range(col+1,n):
            other_queen=board[other_col]
            slope=(queen-other_queen)/(col-other_col)
            if abs(slope)==1:
                score+=1
    return score
def swap(board,num1,num2):
    b=deepcopy(board)
    b[num1]=board[num2]
    b[num2]=board[num1]
    return b
def down_hill(board):
    board_score=fitness_function(board)
    if board_score==0:
        return (board,board_score)
    neighbour=[]
    n=len(board)
    for i in range(n):
        for j in range(i,n):
            neighbour.append((fitness_function(swap(board,i,j)),swap(board,i,j)))
    neighbour.sort()
    x,y=neighbour[0]
    if x>board_score:
        return((board,board_score))
    else:
        return down_hill(y)
# print(down_hill(set_board(8)))
def genetic(generation):
    for score,board in generation:
        print((score,board))
        if score==0:return (board,score)
        neighbour=[]
        n=len(board)
        for i in range(n):
            for j in range(i,n):
                neighbour.append((fitness_function(swap(board,i,j)),swap(board,i,j)))
        neighbour.sort()
        generation.append(neighbour[0])
        generation.append(neighbour[1])
    generation.sort()
    n=int(len(generation)*0.7)
    generation=generation[:n]
    return genetic(generation)
def begin_genetic(size):
    generation=[]
    board=set_board(size)
    generation.append((fitness_function(board),board))
    print(board)
    for i in range(size):
        for j in range(i,size):
            generation.append((fitness_function(swap(board,i,j)),swap(board,i,j)))
    return genetic(generation)
print(begin_genetic(8))
