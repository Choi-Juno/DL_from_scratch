"""
가중치와 편향을 도입하기 전 AND 게이트 구현
----------------------------------
def AND(x1: int, x2: int) -> int:
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1
"""

# 가중치와 편향을 도입한 후의 AND 게이트 구현
import numpy as np


def AND(x1: int, x2: int) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1: int, x2: int) -> int:
    x = np.array([x1, x2])
    # NAND 게이트는 가중치(w)와 편향(b)만 다르다.
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1: int, x2: int) -> int:
    x = np.array([x1, x2])
    # OR 게이트는 가중치(w)와 편향(b)만 다르다.
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1: int, x2: int) -> int:
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


# print(AND(0, 0))
# print(AND(0, 1))
# print(AND(1, 0))
# print(AND(1, 1))
