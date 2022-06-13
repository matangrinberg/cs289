"""
Author: Nathan Miller
Institution: UC Berkeley
Date: Spring 2022
Course: CS189/289A

A Template file for CS 189 Homework 3 question 8.

Feel free to use this if you like, but you are not required to!
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from scipy import io
import pandas as pd
import csv

####################################
# Loading Data
####################################

# MNIST
data_mnist = io.loadmat("data/mnist_data.mat")
n_mnist, n_mnist_test, n_mnist_val, m_mnist = 60000, 10000, 1000, 784
X_train, X_val, y_train, y_val = train_test_split(data_mnist["training_data"],
                                                  data_mnist["training_labels"], test_size=n_mnist_val, random_state=13)
X_train = np.array(X_train).astype(float)
X_val = np.array(X_val).astype(float)
X_test = np.array(data_mnist["test_data"]).astype(float)

# Contranst Normalization, Training Data
# for i in range(n_mnist - n_mnist_val):
#     image = X_train[i]
#     magnitude = np.linalg.norm(image)
#     X_train[i] = X_train[i].astype(float) / magnitude
#
# # Contranst Normalization, Validation Data
# for i in range(n_mnist_val):
#     image = X_val[i]
#     magnitude = np.linalg.norm(image)
#     X_val[i] = X_val[i].astype(float) / magnitude
#
# # Contranst Normalization, Test Data
# for i in range(n_mnist_test):
#     image = X_test[i]
#     magnitude = np.linalg.norm(image)
#     X_test[i] = X_test[i] / magnitude


##############
# GDA
##############

class GDA:
    """Perform Gaussian discriminant analysis (both LDA and QDA)."""
    def __init__(self, *args, **kwargs):
        self._fit = False
        self.means, self.covs, self.avg_cov, self.priors = [], [], [], []

    def evaluate(self, X, y, mode="lda"):
        """Predict and evaluate the accuracy using zero-one loss.

        Args:
            X (np.ndarray): The feature matrix shape (n, d)
            y (np.ndarray): The true labels shape (d,)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            float: The accuracy loss of the learner.

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
        """
        if mode == "3dqda":
            accuracies = []
            predictions = self.predict(X, mode="qda")
            joint_data = np.c_[predictions, y]
            for k in range(9):
                digit_data = np.transpose(joint_data[np.where(joint_data[:, 1].astype(int) == k + 1)])
                accuracy = accuracy_score(digit_data[1], digit_data[0])
                accuracies.append(accuracy)
                print(accuracy)

            return accuracies

        elif mode == "3dlda":
            accuracies = []
            predictions = self.predict(X, mode="lda")
            joint_data = np.c_[predictions, y]
            for k in range(9):
                digit_data = np.transpose(joint_data[np.where(joint_data[:, 1].astype(int) == k + 1)])
                accuracy = accuracy_score(digit_data[1], digit_data[0])
                accuracies.append(accuracy)
                print(accuracy)

            print("mean", np.mean(accuracies))
            return accuracies

        else:
            predictions = self.predict(X, mode=mode)
            accuracy = accuracy_score(y, predictions)
            print(accuracy)
            return accuracy

    def fit(self, x, y):
        """Train the GDA model (both LDA and QDA).

        Args:
            x (np.ndarray): The feature matrix (n, d)
            y (np.ndarray): The true labels (n, d)
        """

        joint_data = np.c_[x, y]
        for k in range(1, 10):
            current_data = joint_data[np.where(joint_data[:, m_mnist].astype(int) == k)][:, 0:m_mnist]
            digit_samples = current_data.shape[0]

            sample_mean = np.mean(current_data, axis=0)
            centered = current_data - sample_mean
            sample_covariance = np.transpose(centered).dot(centered) / digit_samples

            self.means.append(sample_mean)
            self.covs.append(sample_covariance)
            self.priors.append(digit_samples / x.shape[0])

        self.avg_cov = np.sum(self.covs, axis=0) / 9
        self._fit = True

    def predict(self, X, mode="lda"):
        """Use the fitted model to make predictions.

        Args:
            X (np.ndarray): The feature matrix of shape (n, d)

        Optional:
            mode (str): Either "lda" or "qda".

        Returns:
            np.ndarray: The array of predictions of shape (n,)

        Raises:
            RuntimeError: If an unknown mode is passed into the method.
            RuntimeError: If called before model is trained
        """
        if not self._fit:
            raise RuntimeError("Cannot predict for a model before `fit` is called")

        preds = None
        if mode == "lda":
            discriminants = np.zeros([9, X.shape[0]])
            for k in range(9):
                # discriminants[k] = multivariate_normal.logpdf(X, self.means[k], self.avg_cov, allow_singular=1) + np.log(self.priors[k])
                discriminants[k] = multivariate_normal.logpdf(X, self.means[k], self.avg_cov, allow_singular=1)
            preds = np.argmax(discriminants, axis=0) + 1
        elif mode == "qda":
            discriminants = np.zeros([9, X.shape[0]])
            for k in range(9):
                discriminants[k] = multivariate_normal.logpdf(X, self.means[k], self.covs[k], allow_singular=1) + np.log(self.priors[k])
            preds = np.argmax(discriminants, axis=0) + 1
        else:
            raise RuntimeError("Unknown mode!")
        return preds


sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]

# Part 3A
# errors = []
# for size in sizes:
#     g = GDA()
#     g.fit(X_train[0:size], y_train[0:size])
#     errors.append(1 - g.evaluate(X_val, y_val, mode="lda"))
# plt.plot(sizes, errors)
# plt.xlabel("# of training points")
# plt.ylabel("Error rate")
# plt.ylim([0, 1])
# plt.title("MNIST LDA classificiation")
# plt.show()


# Part 3B
# errors = []
# for size in sizes:
#     g = GDA()
#     g.fit(X_train[0:size], y_train[0:size])
#     errors.append(1 - g.evaluate(X_val, y_val, mode="qda"))
#
# plt.plot(sizes, errors)
# plt.xlabel("# of training points")
# plt.ylabel("Error rate")
# plt.ylim([0, 1])
# plt.title("MNIST QDA classificiation")
# plt.show()

# Part 3D, LDA
errors = []
scores = []
for size in sizes:
    g = GDA()
    g.fit(X_train[0:size], y_train[0:size])
    scores = g.evaluate(X_val, y_val, mode="3dlda")
    error = np.array([1 - score for score in scores])
    errors.append(error)

errors = np.transpose(errors)

for i in range(9):
    plt.plot(sizes, errors[i])

plt.xlabel("# of training points")
plt.ylabel("Error rate")
plt.ylim([0, 1])
plt.title("MNIST LDA classificiation, digitwise")
plt.legend(["Digit 1", "Digit 2", "Digit 3", "Digit 4", "Digit 5", "Digit 6", "Digit 7", "Digit 8", "Digit 9"])
plt.show()

# Part 3D, QDA
# errors = []
# scores = []
# for size in sizes:
#     g = GDA()
#     g.fit(X_train[0:size], y_train[0:size])
#     scores = g.evaluate(X_val, y_val, mode="3dqda")
#     error = np.array([1 - score for score in scores])
#     errors.append(error)
#
# errors = np.transpose(errors)
#
# for i in range(9):
#     plt.plot(sizes, errors[i])
#
# plt.xlabel("# of training points")
# plt.ylabel("Error rate")
# plt.ylim([0, 1])
# plt.title("MNIST QDA classificiation, digitwise")
# plt.legend(["Digit 1", "Digit 2", "Digit 3", "Digit 4", "Digit 5", "Digit 6", "Digit 7", "Digit 8", "Digit 9"])
# plt.show()


# Part 4

# size = 59000
# g = GDA()
# g.fit(X_train[0:size], y_train[0:size])
# validation_predictions = g.predict(X_val)
# score = g.evaluate(X_val, y_val, mode="lda")
# print(score)


# def results_to_csv(y_test):
#     y_test = y_test.astype(int)
#     df = pd.DataFrame({'Category': y_test})
#     df.index += 1  # Ensures that the index starts at 1.
#     df.to_csv('mnist_submission.csv', index_label='Id')
#
#
# test_predictions = g.predict(X_test)
# results_to_csv(test_predictions)

