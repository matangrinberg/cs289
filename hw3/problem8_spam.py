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

# SPAM
data_spam = io.loadmat("data/spam_data.mat")
n_spam, n_spam_test, n_spam_val, m_spam = 5172, 5857, 500, 32
X_train, X_val, y_train, y_val = train_test_split(data_spam["training_data"],
                                                  data_spam["training_labels"], test_size=n_spam_val, random_state=123)

X_train = np.array(X_train).astype(float)
X_val = np.array(X_val).astype(float)
X_test = np.array(data_spam["test_data"]).astype(float)


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
        for k in range(2):
            current_data = joint_data[np.where(joint_data[:, m_spam].astype(int) == k)][:, 0:m_spam]
            digit_samples = current_data.shape[0]

            sample_mean = np.mean(current_data, axis=0)
            centered = current_data - sample_mean
            sample_covariance = np.transpose(centered).dot(centered) / digit_samples

            self.means.append(sample_mean)
            self.covs.append(sample_covariance)
            self.priors.append(digit_samples / x.shape[0])

        self.avg_cov = np.sum(self.covs, axis=0) / 2
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
            discriminants = np.zeros([2, X.shape[0]])
            for k in range(2):
                print(k)
                discriminants[k] = multivariate_normal.logpdf(X, self.means[k], self.avg_cov, allow_singular=1) \
                                   + np.log(self.priors[0])

            preds = np.argmax(discriminants, axis=0)
            print(preds)
        elif mode == "qda":
            discriminants = np.zeros([2, X.shape[0]])
            for k in range(2):
                discriminants[k] = multivariate_normal.logpdf(X, self.means[k], self.covs[k], allow_singular=1) \
                                   + np.log(self.priors[k])
            preds = np.argmax(discriminants, axis=0)
        else:
            raise RuntimeError("Unknown mode!")
        return preds


size = 4500
g = GDA()
g.fit(X_train[0:size], y_train[0:size])
validation_predictions = g.predict(X_val)
score = g.evaluate(X_val, y_val, mode="lda")
print(score)


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('spam_submission.csv', index_label='Id')


test_predictions = g.predict(X_test)
results_to_csv(test_predictions)

