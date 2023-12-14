import random


def row_win(board):
    for i in range(3):
        if board[i][0] == board[i][2] and board[i][1] == board[i][2]:
            return board[i][1]


def change_player(player):
    if player == "x":
        return "o"
    else:
        return "x"


def col_win(board):
    for i in range(3):
        if board[0][i] == board[1][i] and board[1][i] == board[2][i]:
            return board[1][i]


def diag_win(board):
    if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        return board[1][1]
    if board[0][2] == board[1][1] and board[1][1] == board[2][0]:
        return board[1][1]


def win(board):
    if row_win(board):
        return row_win(board)
    if col_win(board):
        return col_win(board)
    if diag_win(board):
        return diag_win(board)


def is_tie(board):
    if win(board):
        return False
    for i in range(3):
        for j in range(3):
            if not (board[j][i]):
                return False
    return True


def open_spaces(board):
    spaces = []
    for i in range(3):
        for j in range(3):
            if not (board[j][i]):
                spaces.append((i, j))
    return spaces


def is_over(board):
    if win(board):
        return True
    if is_tie(board):
        return True
    return False


def score(board, player):
    winner = win(board)
    if winner == "x":
        return 1
    elif winner == "o":
        return -1
    actions = open_spaces(board)
    if is_tie(board):
        return 0
    children = []
    next_player = change_player(player)
    for x, y in actions:
        b = deepcopy(board)
        b[y][x] = player
        children.append(score(b, next_player))
    random.shuffle(children)
    if player == "o":
        return min(children)
    else:
        return max(children)


def get_winner(board, prev_player):  # every body plays best
    winner = win(board)
    if winner == "x":
        return 1
    elif winner == "o":
        return -1

    current_player = change_player(prev_player)
    child_scores = []

    for empty_row, empty_col in open_spaces(board):
        child_board = deepcopy(board)
        child_board[empty_col][empty_row] = current_player

        child_score = get_winner(child_board, current_player)
        child_scores.append(child_score)

    if len(child_scores) == 0:
        return 0
    elif current_player == "x":
        return max(child_scores)
    elif current_player == "o":
        return min(child_scores)


from copy import deepcopy


def smart_player(board, player):
    next_player = change_player(player)
    children = []
    actions = open_spaces(board)
    if not actions:
        return
    for action in actions:
        b = deepcopy(board)
        x, y = action
        b[y][x] = player
        children.append((get_winner(b, player), action))
    priority = []
    my_value, my_action = children[0]
    for child in children:
        x, y = child
        if player == "x":
            if x > my_value:
                my_value = x
                my_action = y

        if player == "o":
            if x < my_value:
                my_value = x
                my_action = y
    return my_action


def smart_playerh(board, player):
    next_player = change_player(player)
    children = []
    actions = open_spaces(board)
    if not actions:
        return
    for action in actions:
        b = deepcopy(board)
        x, y = action
        b[y][x] = player
        children.append((score(b, next_player), action))
    priority = []
    my_value, my_action = children[0]
    for child in children:
        x, y = child
        if player == "x":
            if x > my_value:
                my_value = x
                my_action = y

        if player == "o":
            if x < my_value:
                my_value = x
                my_action = y
    return my_action


def rand_player(board, player):
    a = open_spaces(board)
    random.shuffle(a)
    return a[0]


def test():
    board = [["o", "", "x"], ["", "o", "o"], ["x", "", "x"]]

    player = "x"
    print(rand_player(board, player))


def move(board, act, player):
    x, y = act
    board[y][x] = player


def game(
    board=[["", "", ""], ["", "", ""], ["", "", ""]], turn="g", player="x"
):  # the game between random player and smart
    if turn == "g":
        turn = random.choice([True, False])
        print("did the yz player start?")
        print(turn)
    if win(board):
        if turn:
            return -1
        else:
            return 1
    if is_tie(board):
        return 0
    if turn:
        act = smart_player(board, player)
        move(board, act, player)
    else:
        act = smart_playerh(board, player)
        move(board, act, player)
    print(board)
    turn = not turn
    player = change_player(player)
    return game(board, turn, player)

print(game())
