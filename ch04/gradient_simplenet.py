import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient
import numpy as np


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])


def f(W):
    return net.loss(x, t)
