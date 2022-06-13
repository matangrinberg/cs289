
import numpy as np
from scipy.special import expit

X = np.array([[0.2, 3.1, 1.0], [1.0, 3.0, 1.0], [-0.2, 1.2, 1.0], [1.0, 1.1, 1.0]])
y = np.array([[1], [1], [0], [0]])
w0 = [[-1.0], [1.0], [0.0]]
s0 = expit(X.dot(w0))


def newton(w, s, x, d):
    pseudo_inv = np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x))
    end = ((s * (1 - s)) ** -1) * (s - d)
    return w - pseudo_inv.dot(end)


w1 = newton(w0, s0, X, y)
s1 = expit(X.dot(w1))
w2 = newton(w1, s1, X, y)
s2 = expit(X.dot(w2))
