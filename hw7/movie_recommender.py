import os
import scipy.io
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Load training data from MAT file
R_data = scipy.io.loadmat('movie_data/movie_train.mat')['train']

# Load validation data from CSV
val_data = np.loadtxt('movie_data/movie_validate.txt', dtype=int, delimiter=',')


# Method to get training accuracy
def get_train_acc(R, user_vecs, movie_vecs):
    num_correct, total = 0, 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if not np.isnan(R[i, j]):
                total += 1
                if np.dot(user_vecs[i], movie_vecs[j])*R[i, j] > 0:
                    num_correct += 1
    return num_correct/total


# Method to get validation accuracy
def get_val_acc(val_data, user_vecs, movie_vecs):
    num_correct = 0
    for val_pt in val_data:
        user_vec = user_vecs[val_pt[0]-1]
        movie_vec = movie_vecs[val_pt[1]-1]
        est_rating = np.dot(user_vec, movie_vec)
        if est_rating*val_pt[2] > 0:
            num_correct += 1
    return num_correct/val_data.shape[0]


# Method to get indices of all rated movies for each user,
# and indices of all users who have rated that title for each movie
def get_rated_idxs(R):
    user_rated_idxs, movie_rated_idxs = [], []
    for i in range(R.shape[0]):
        user_rated_idxs.append(np.argwhere(~np.isnan(R[i, :])).reshape(-1))
    for j in range(R.shape[1]):
        movie_rated_idxs.append(np.argwhere(~np.isnan(R[:, j])).reshape(-1))
    return np.array(user_rated_idxs), np.array(movie_rated_idxs)


# Part (c): SVD to learn low-dimensional vector representations
def svd_lfm(R):
    # Fill in the missing values in R
    R = np.nan_to_num(R)

    # Compute the SVD of R
    user_vecs, D_vec, movie_vecs = scipy.linalg.svd(R, full_matrices=False)

    # Construct user and movie representations
    user_vecs = user_vecs * D_vec
    movie_vecs = np.transpose(movie_vecs)

    return user_vecs, movie_vecs


# Part (d): Compute the training MSE loss of a given vectorization
def get_train_mse(R, user_vecs, movie_vecs):

    # Compute the training MSE loss
    mse_loss, count = 0, 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isnan(R[i, j]):
                pass
            else:
                mse_loss += (np.dot(user_vecs[i], movie_vecs[j]) - R[i, j]) ** 2
                count += 1

    return mse_loss/count


# Part (e): Compute training MSE and val acc of SVD LFM for various d
# d_values = [2, 5, 10, 20]
# train_mses, train_accs, val_accs = [], [], []
# user_vecs, movie_vecs = svd_lfm(np.copy(R_data))
# for d in d_values:
#     print('Checking d_value:', d)
#     train_mses.append(get_train_mse(np.copy(R_data), user_vecs[:, :d], movie_vecs[:, :d]))
#     train_accs.append(get_train_acc(np.copy(R_data), user_vecs[:, :d], movie_vecs[:, :d]))
#     val_accs.append(get_val_acc(val_data, user_vecs[:, :d], movie_vecs[:, :d]))
# plt.clf()
# plt.plot([str(d) for d in d_values], train_mses, 'o-')
# plt.title('Train MSE of SVD-LFM with Varying Dimensionality')
# plt.xlabel('d')
# plt.ylabel('Train MSE')
# plt.savefig(fname='train_mses.png', dpi=600, bbox_inches='tight')
# plt.clf()
# plt.plot([str(d) for d in d_values], train_accs, 'o-')
# plt.plot([str(d) for d in d_values], val_accs, 'o-')
# plt.title('Train/Val Accuracy of SVD-LFM with Varying Dimensionality')
# plt.xlabel('d')
# plt.ylabel('Train/Val Accuracy')
# plt.legend(['Train Accuracy', 'Validation Accuracy'])
# plt.savefig(fname='trval_accs.png', dpi=600, bbox_inches='tight')


####################################################################################

# Part (f): Learn better user/movie vector representations by minimizing loss
best_d = 10
np.random.seed(20)
user_vecs = np.random.random((R_data.shape[0], best_d))
movie_vecs = np.random.random((R_data.shape[1], best_d))
user_rated_idxs, movie_rated_idxs = get_rated_idxs(np.copy(R_data))


# Part (f): Update USER (x_i) vectors
def update_user_vecs(user_vecs, movie_vecs, R, user_rated_idxs):
    # Update user_vecs to the loss-minimizing value
    n, m, d = user_vecs.shape[0], movie_vecs.shape[0], user_vecs.shape[1]
    for k in range(n):
        y_mat, Ry_vec = np.identity(d), np.zeros(d)
        for index in user_rated_idxs[k]:
            y_vec = movie_vecs[index]
            y_mat += np.outer(y_vec, y_vec)
            Ry_vec += R[k, index] * y_vec

        user_vecs[k] = np.matmul(np.linalg.inv(y_mat), Ry_vec)

    return user_vecs


# Part (f): Update MOVIE (y_i) vectors
def update_movie_vecs(user_vecs, movie_vecs, R, movie_rated_idxs):
    # Update movie_vecs to the loss-minimizing value
    n, m, d = user_vecs.shape[0], movie_vecs.shape[0], movie_vecs.shape[1]
    for k in range(m):
        x_mat, xR_vec = np.identity(d), np.zeros(d)
        for index in movie_rated_idxs[k]:
            x_vec = user_vecs[index]
            x_mat += np.outer(x_vec, x_vec)
            xR_vec += x_vec * R[index, k]

        movie_vecs[k] = np.matmul(xR_vec, np.linalg.inv(x_mat))

    return movie_vecs


# Part (f): Perform loss optimization using alternating updates
train_mse = get_train_mse(np.copy(R_data), user_vecs, movie_vecs)
train_acc = get_train_acc(np.copy(R_data), user_vecs, movie_vecs)
val_acc = get_val_acc(val_data, user_vecs, movie_vecs)
print(f'Start optim, train MSE: {train_mse:.2f}, train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}')
for opt_iter in range(20):
    user_vecs = update_user_vecs(user_vecs, movie_vecs, np.copy(R_data), user_rated_idxs)
    movie_vecs = update_movie_vecs(user_vecs, movie_vecs, np.copy(R_data), movie_rated_idxs)
    train_mse = get_train_mse(np.copy(R_data), user_vecs, movie_vecs)
    train_acc = get_train_acc(np.copy(R_data), user_vecs, movie_vecs)
    val_acc = get_val_acc(val_data, user_vecs, movie_vecs)
    print(f'Iteration {opt_iter+1}, train MSE: {train_mse:.2f}, train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}')

