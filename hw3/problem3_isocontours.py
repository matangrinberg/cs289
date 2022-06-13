import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

#########
# Plots
#########

# Problem 1


mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0], [0, 2]])

x1, y1 = np.mgrid[-4:4:.01, -5:5:.01]
pos1 = np.dstack((x1, y1))
pdf1 = multivariate_normal(mu1, sigma1)
z1 = pdf1.pdf(pos1)

plt.contourf(x1, y1, z1, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('3.1 Isocontours')
plt.show()

# Problem 2

mu2 = np.array([-1, 2])
sigma2 = np.array([[2, 1], [1, 4]])

x2, y2 = np.mgrid[-4:4:.01, -5:5:.01]
pos2 = np.dstack((x2, y2))
pdf2 = multivariate_normal(mu2, sigma2)
z2 = pdf2.pdf(pos2)

plt.contourf(x2, y2, z2, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('3.2 Isocontours')
plt.show()

# Problem 3

mu3a, mu3b = np.array([0, 2]), np.array([2, 0])
sigma3a, sigma3b = np.array([[2, 1], [1, 1]]), np.array([[2, 1], [1, 1]])

x3, y3 = np.mgrid[-5:5:.01, -5:5:.01]
pos3 = np.dstack((x3, y3))
pdf3a = multivariate_normal(mu3a, sigma3a)
pdf3b = multivariate_normal(mu3b, sigma3b)
z3 = pdf3a.pdf(pos3) - pdf3b.pdf(pos3)

plt.contourf(x3, y3, z3, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('3.3 Isocontours')
plt.show()

# Problem 4

mu4a, mu4b = [0, 2], [2, 0]
sigma4a, sigma4b = [[2, 1], [1, 1]], [[2, 1], [1, 4]]

x4, y4 = np.mgrid[-5:5:.01, -5:5:.01]
pos4 = np.dstack((x4, y4))
pdf4a = multivariate_normal(mu4a, sigma4a)
pdf4b = multivariate_normal(mu4b, sigma4b)
z4 = pdf4a.pdf(pos4) - pdf4b.pdf(pos4)

plt.contourf(x4, y4, z4, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('3.4 Isocontours')
plt.show()

# Problem 5

mu5a, mu5b = [1, 1], [-1, -1]
sigma5a, sigma5b = [[2, 0], [0, 1]], [[2, 1], [1, 2]]

x5, y5 = np.mgrid[-5:5:.01, -5:5:.01]
pos5 = np.dstack((x5, y5))
pdf5a = multivariate_normal(mu5a, sigma5a)
pdf5b = multivariate_normal(mu5b, sigma5b)
z5 = pdf5a.pdf(pos5) - pdf5b.pdf(pos5)

plt.contourf(x5, y5, z5, 20, cmap='viridis')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('3.5 Isocontours')
plt.show()

