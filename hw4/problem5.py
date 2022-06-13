
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import expit
from scipy import io
import pandas as pd
import csv
import random
np.random.seed(12345)


def norm(vec, p):
    if p == 0:
        return 1
    # if p == 1:
    #     return np.sum(np.abs(vec), axis=2)
    return np.power(np.sum(np.power(np.abs(vec), p), axis=2), 1/p)


#########
# Part 1
#########
# (a)
p0 = 1/2
w1, w2 = np.mgrid[-2:2:.05, -2:2:.05]
points = np.dstack((w1, w2))
norms = norm(points, p0)

plt.contourf(w1, w2, norms, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('5.1a Isocontours of $\ell_{0.5}$')
plt.savefig('1.png')
plt.show()

# (b)
p0 = 1
w1, w2 = np.mgrid[-2:2:.05, -2:2:.05]
points = np.dstack((w1, w2))
norms = norm(points, p0)

plt.contourf(w1, w2, norms, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('5.1b Isocontours of $\ell_{1}$')
plt.savefig('2.png')
plt.show()

# (c)
p0 = 2
w1, w2 = np.mgrid[-2:2:.05, -2:2:.05]
points = np.dstack((w1, w2))
norms = norm(points, p0)

plt.contourf(w1, w2, norms, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('5.1c Isocontours of $\ell_{2}$')
plt.savefig('3.png')
plt.show()


