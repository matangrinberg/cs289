
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

####################################
# Loading and Normalizing Data
####################################

# Loading
data_wine = io.loadmat('data_wine.mat')
n, n_test, n_val, m = 6000, 497, 500, 12
n_train = n - n_val
y, X, X_test, desc = data_wine['y'], data_wine['X'], data_wine['X_test'], data_wine['description']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_val, random_state=12345)

# Normalization
means, stds = [], []
for i in range(m):
    mu, std = np.mean(X_train[:, i]), np.std(X_train[:, i])
    means.append(mu)
    stds.append(std)
    X_train[:, i] = (X_train[:, i] - means[i]) / stds[i]
    X_val[:, i] = (X_val[:, i] - means[i]) / stds[i]
    X_test[:, i] = (X_test[:, i] - means[i]) / stds[i]

# Adding Fictitious Dimension
X_train, X_val = np.c_[X_train, np.ones(n - n_val)], np.c_[X_val, np.ones(n_val)]
X_test = np.c_[X_test, np.ones(n_test)]


def cost(x_data, y_data, weights, reg_param):
    eps = 0.00000000000000001
    s = expit(x_data.dot(weights))
    reg = reg_param / 2 * (np.linalg.norm(weights[:12]) ** 2)
    logistic = - np.squeeze(y_data).dot(np.log(s + eps)) - np.squeeze(1 - y_data).dot(np.log(1 - s + eps))
    return reg + logistic


#########################################
# Part 2, Batch Gradient Descent Code
#########################################

def GDupdate(x_data, y_data, weights, reg_param, step_size):
    s = expit(x_data.dot(weights))
    res = - step_size * (np.transpose(x_data).dot(np.squeeze(y_data) - s) - reg_param * weights)
    return res


e, l, iterations = 0.001, 0.000001, 10000
w = [np.zeros(13)]

costs = np.zeros(iterations + 1)
costs[0] = cost(X_train, y_train, w[0], l)


for i in range(0, iterations):
    w_new = w[i] - GDupdate(X_train, y_train, w[i], l, e)
    w.append(w_new)
    costs[i + 1] = cost(X_train, y_train, w_new, l)

print(costs)
#
# plt.plot(np.log(range(1, iterations + 2)), costs)
# plt.xlabel('$\ln($ Iterations $)$')
# plt.ylabel('Cost Function')
# plt.title('BGD: Cost vs. Iterations, $\epsilon=0.001$, $\lambda = 0.000001$')
# plt.show()


############################################
# Part 4, Stochastic Gradient Descent Code
############################################
# e, l, iterations = 0.02, 0.0000005, 500000
# w = [np.zeros(13)]
#
#
# def SGDupdate(x_point, y_point, weights, reg_param, step_size):
#     s = expit(x_point.dot(weights))
#     res = - step_size * (np.transpose(x_point).dot(y_point[0] - s) - reg_param * weights)
#     return res
#
#
# costs = np.zeros(iterations)
# costs[0] = cost(X_train, y_train, w[0], l)
# permutations = np.random.permutation(n_train)
#
# for i in range(0, iterations):
#     if i % n_train == n_train - 1:
#         permutations = np.random.permutation(n_train)
#     index = permutations[i % n_train]
#     w_new = w[i] - SGDupdate(X_train[index], y_train[index], w[i], l, e)
#     w.append(w_new)
#     costs[i] = cost(X_train, y_train, w_new, l)
#
# print(costs)
#
# plt.plot(np.log(range(1, iterations + 1)), costs)
# plt.xlabel('$\ln(Iterations)$')
# plt.ylabel('Cost Function')
# plt.title('SGD: Cost vs. Iterations, $\epsilon=0.02$, $\lambda = 0.0000005$')
# plt.show()


############################################
# Part 5, Varying SGD
############################################
# e, l, iterations = 300, 0.001, 200000
# w = [np.zeros(13)]
#
#
# def SGDupdate(x_point, y_point, weights, reg_param, step_size):
#     s = expit(x_point.dot(weights))
#     res = - step_size * (np.transpose(x_point).dot(y_point[0] - s) - reg_param * weights)
#     return res
#
#
# costs = np.zeros(iterations)
# costs[0] = cost(X_train, y_train, w[0], l)
# permutations = np.random.permutation(n_train)
#
# for i in range(0, iterations):
#     if i % n_train == n_train - 1:
#         permutations = np.random.permutation(n_train)
#     index = permutations[i % n_train]
#     w_new = w[i] - SGDupdate(X_train[index], y_train[index], w[i], l, e) / (i + 1)
#     w.append(w_new)
#     costs[i] = cost(X_train, y_train, w_new, l)
#
# print(costs)
#
# # plt.plot(np.log(range(1, iterations + 1)), costs)
# plt.plot(np.log(range(1, iterations + 1)), costs)
# plt.xlabel('$\ln(Iterations)$')
# plt.ylabel('Cost Function')
# plt.title('Varying SGD: Cost vs. Iterations, $\delta = 2000$, $\lambda = 0.0003$')
# plt.show()


############################################
# Part 6, Test Prediction
############################################

def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('wine_submission.csv', index_label='Id')


# Validation
w_best = w[-1]
predictions = np.round(expit(X_val.dot(w_best)))
score = accuracy_score(y_val, predictions)
print(score)

# Test
test_predictions = np.round(expit(X_test.dot(w_best)))
results_to_csv(test_predictions)





