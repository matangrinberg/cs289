import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from scipy import io

# Part 1

####################################
# Loading Data
####################################

data_mnist = io.loadmat("data/mnist_data.mat")
data_spam = io.loadmat("data/spam_data.mat")
fields = "test_data", "training_data", "training_labels"
for field in fields:
    print(field, data_mnist[field].shape)
    print(field, data_spam[field].shape)

n_mnist, n_mnist_test, m_mnist = 60000, 10000, 784
n_spam, n_spam_test, m_spam = 5172, 5857, 32


mnist_training = np.c_[np.array(data_mnist["training_data"]).astype(float), np.array(data_mnist["training_labels"])]
mnist_training_sorted = mnist_training[mnist_training[:, m_mnist].argsort()]
mnist_test = np.array(data_mnist["test_data"])

####################################
# Contranst Normalization
####################################

for i in range(n_mnist):
    image = mnist_training_sorted[i, 0:m_mnist]
    magnitude = np.linalg.norm(image)
    mnist_training_sorted[i, 0:m_mnist] = mnist_training_sorted[i, 0:m_mnist] / magnitude


#############################################
# Fitting Gaussian using MLE
#############################################
means = []
covariances = []
for i in range(1, 10):
    current_data = mnist_training_sorted[np.where(mnist_training_sorted[:, 784].astype(int) == i)][:, 0:784]
    digit_samples = current_data.shape[0]

    sample_mean = np.mean(current_data, axis=0)
    centered = current_data - sample_mean
    sample_covariance = np.transpose(centered).dot(centered) / digit_samples

    means.append(sample_mean)
    covariances.append(sample_covariance)

# We now have the sample mean and sample covariance for each digit class.
# To turn this into a Gaussian distribution we simply plug them into multivariate_normal


###############
# Part 2
###############
# We are considering the digit 9.

# Mean
plt.imshow(np.reshape(sample_mean, (28, 28)))
plt.colorbar()
plt.show()

# Covariance
plt.imshow(sample_covariance)
plt.colorbar()
plt.show()


