import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

np.random.seed(12345)

n = 100
m1, m2, s1, s2 = 3, 4, 3, 2
x1 = np.random.normal(m1, s1, n)
x2 = x1 / 2 + np.random.normal(m2, s2, n)
X = np.array([x1, x2])


# Part 1

mu = np.array([np.mean(x1), np.mean(x2)])
print(mu)

# Part 2

cov = np.cov(X)
print(cov)

# Part 3

vals, vecs = np.linalg.eig(cov)
print(np.linalg.eig(cov))

# Part 4

# plt.scatter(x1, x2)
# plt.xlim(-15, 15)
# plt.ylim(-15, 15)
# plt.scatter(mu[0], mu[1], c='red')
# plt.quiver(mu[0], mu[1], vecs[0][0], vecs[1][0], scale=30/vals[0])
# plt.quiver(mu[0], mu[1], vecs[0][1], vecs[1][1], scale=30/vals[1])
# plt.title('Scatter Plot for X1 and X2 with Eigenvectors Scaled by Eigenvalues')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# Part 5

u = vecs
u_t = np.transpose(u)
X_c = np.transpose(np.transpose(X) - mu)
X_cr = u_t.dot(X_c)

plt.scatter(X_cr[0], X_cr[1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(0, 0, c='red')
plt.title('Scatter Plot of Centered and Rotated Data')
plt.xlabel('$U^T(X_1-\mu_1)$')
plt.ylabel('$U^T(X_2-\mu_2)$')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





