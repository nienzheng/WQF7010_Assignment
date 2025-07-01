import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from properties import GlobalProperties
# from math import inf as infinity
from copy import deepcopy

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
    # plt.imshow(dltd, cmap='gray')
    # plt.title("Dilated Image")
    # plt.axis('off')
    # plt.show()
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

def init():
    global cap
    cap = cv2.VideoCapture(0)
    print("Video initialized {0}x{1}, {2} fps".format(int(cap.get(3)), int(cap.get(4)), int(cap.get(5))))

def _paste_non_zero(dest, src):
    s = deepcopy(dest)
    for i in range(len(s)):
        for j in range(len(s[i])):
            if np.any(src[i][j] != 0):
                s[i][j] = src[i][j]
    return s


def _add_lines(frame, lines):
    sh = frame.shape
    for i in lines[0]:
        cv2.line(frame, (0, i), (sh[1], i), (0, 255, 0))

    for i in lines[1]:
        cv2.line(frame, (i, 0), (i, sh[0]), (0, 255, 0))


def get_state():
    threshold = 5
    # for testing
    # frame = cv2.imread('out3.png')
    ret, frame = cap.read()
    binary = to_binary_color(frame)
    sq, lines = split_to_squares(binary)
    if len(sq) < 2 or len(sq) > threshold or len(sq[0]) > threshold:
        cv2.imshow('frame', binary)
        cv2.waitKey(1) & 0xFF == ord('q')
        return []
    state = []
    # ip.recognize_shape(sq[1][1])
    for row in sq:
        state.append([])
        for column in row:
            state[-1].append(recognize_shape(column))
    # cv2.imwrite('out4.png', binary)
    _add_lines(binary, lines)
    cv2.imshow('frame', binary)
    cv2.waitKey(1) & 0xFF == ord('q')  # required to show
    return state


def destroy():
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()