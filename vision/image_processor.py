import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from properties import GlobalProperties
from math import inf as infinity

HUMAN = -1
COMP = +1

def main_cgpt():
    global hkernel
    global vkernel
    global horizontal_rhos
    global vertical_rhos
    
    # 1. Load the image
    image = cv2.imread('s0.png')
    if image is None:
        raise FileNotFoundError("Image not found: out2.png")
    
    binary = to_binary_color(image)
    
    # 3. Create horizontal and vertical kernels
    hkernel = np.ones((1, 100), np.uint8)  # for horizontal lines
    vkernel = np.ones((100, 1), np.uint8)  # for vertical lines
    
    # 4. Detect and filter lines
    horizontal_rhos = _get_lines(binary, hkernel)
    vertical_rhos = _get_lines(binary, vkernel)
    
    print('horizontal_rhos:',horizontal_rhos)
    print('vertical_rhos:',vertical_rhos)
    
    filtered_horizontal = _filter_lines(horizontal_rhos)
    filtered_vertical = _filter_lines(vertical_rhos)
    
    # 5. Print the detected line positions
    print("Horizontal lines (y-coordinates):", filtered_horizontal)
    print("Vertical lines (x-coordinates):", filtered_vertical)

def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False

def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, HUMAN) or wins(state, COMP)


def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells

def evaluate(state):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score

def minimax(state, depth, player):
    
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best

def main():
    global squares
    global hlines
    global vlines
    global binary
    global state
    global x
    global y
    
    # board = [
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    # ]
    
    # 1. Load the image
    image = cv2.imread('s9a.png')
    if image is None:
        raise FileNotFoundError("Image not found: out2.png")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    squares, (hlines, vlines) = split_to_squares(binary)
    
    rows = len(squares)
    cols = len(squares[0]) if rows > 0 else 0
    
    plt.figure(figsize=(cols * 2, rows * 2))  # adjust size as needed
    
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, i * cols + j + 1)
            plt.imshow(squares[i][j], cmap='gray')
            plt.axis('off')
            plt.title(f'[{i},{j}]')
    
    plt.suptitle("Grid Cells")
    plt.tight_layout()
    plt.show()
    
    state = []
    for row in squares:
        state.append([])
        for column in row:
            state[-1].append(recognize_shape(column))
    
    # depth = len(empty_cells(state))
    # move = minimax(state, depth, COMP)
    # x, y = move[0], move[1]
    # print(x,y)
    
    print(f"Detected {len(hlines)-1} rows and {len(vlines)-1} columns.")

def to_binary_color(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, GlobalProperties.black_threshold, 255, cv2.THRESH_BINARY)
    return binary

def _get_lines(frame, kernel):
    global dltd
    global lines
    dltd = cv2.dilate(frame, kernel, iterations=1)
    plt.imshow(dltd, cmap='gray')
    plt.title("Dilated Image")
    plt.axis('off')
    plt.show()
    lines = cv2.HoughLines(255 - dltd, 1, np.pi / 180, GlobalProperties.line_votes)
    return [] if lines is None else np.array([abs(l[0][0]) for l in lines])


def _filter_lines(lines):
    # threshold = GlobalProperties.line_space_threshold
    threshold = 200
    res = []
    for l in lines:
        take = True
        for lr in res:
            if abs(l - lr) < threshold:
                take = False

        if take:
            res.append(int(l))
    print(res)
    return res


def split_to_squares(frame):
    hkernel = np.ones((1, 100))
    vkernel = np.ones((100, 1))
    hlines = [0] + _filter_lines(_get_lines(frame, hkernel)) + [frame.shape[0] - 1]
    vlines = [0] + _filter_lines(_get_lines(frame, vkernel)) + [frame.shape[1] - 1]
    hlines.sort()
    vlines.sort()
    res = []
    for i in range(1, len(hlines)):
        res.append([])
        for j in range(1, len(vlines)):
            res[-1].append(frame[hlines[i - 1]:hlines[i], vlines[j - 1]:vlines[j]])
    return res, (hlines, vlines)


def _is_circle(frame):
    if np.all(frame == 0):
        return False
    circ = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=25, minRadius=0, maxRadius=0)
    return circ is not None and len(circ) == 1


def _pad_image(frame, padding=40):
    sh = frame.shape
    return frame[padding:sh[0] - padding, padding:sh[1] - padding]


def _is_empty(frame):
    threshold = 50
    padded = _pad_image(frame)
    return np.count_nonzero(255 - padded) < threshold


def _filter_cross_lines(lines):
    if lines is None:
        return []
    threshold = 0.35
    min = GlobalProperties.min_cross_angle  # minimal angle
    max = GlobalProperties.max_cross_angle  # maximal angle
    res = []
    for l in lines:
        ang = l[0][1] if l[0][1] < math.pi/2 else math.pi - l[0][1]
        if ang < min or ang > max:
            continue
        toadd = True
        for r in res:
            if abs(r - l[0][1]) < threshold:
                toadd = False
        if toadd:
            res.append(l[0][1])
    return res


def _is_cross(frame):
    if np.all(frame == 0):
        return False
    pframe = _pad_image(frame, 20)
    lines = cv2.HoughLines(255 - pframe, 1, np.pi / 180, 120)
    # cv2.imshow('frame2', pframe)
    # print(lines)
    # print(_filter_cross_lines(lines))
    return len(_filter_cross_lines(lines)) == 2

# def recognize_shape(frame):
#     if _is_cross(frame):
#         return 1
#     if _is_circle(frame):
#         return -1
#     return 0
def recognize_shape(frame):
    if _is_cross(frame):
        return -1
    if _is_circle(frame):
        return 1
    return 0
# def recognize_shape(frame):
#     if _is_cross(frame):
#         return 'x'
#     if _is_circle(frame):
#         return 'o'
#     return 0

if __name__ == "__main__":
    main()

