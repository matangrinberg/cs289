
from collections import Counter
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import sklearn.tree
import random
import pandas as pd
import matplotlib.pyplot as plt

random.seed(12344)
np.random.seed(12344)

eps = 1e-5  # a small number


# Vectorized function for hashing for np efficiency
def w(x):
    return np.int(hash(x)) % 1000


h = np.vectorize(w)


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, left=None, right=None, split_rule=None, label=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left = left
        self.right = right
        self.split_rule = split_rule
        self.label = label
        self.predict = np.vectorize(self.scalar_predict, signature='(n)->()')

    @staticmethod
    def information_gain(X, y, thresh):
        # Initial Entropy
        p = np.array(list(Counter(y).values())) / len(y)
        logp = np.log(p) / np.log(2)
        ent = np.sum(- p * logp)

        # Splitting
        left_y, right_y = [], []
        for i in range(X.shape[0]):
            if X[i] <= thresh:
                left_y.append(y[i])
            else:
                right_y.append(y[i])

        # Final Entropy
        l, r = len(left_y), len(right_y)
        p_l, p_r = np.array(list(Counter(left_y).values())) / l, np.array(list(Counter(right_y).values())) / r
        logp_l, logp_r = np.log(p_l) / np.log(2), np.log(p_r) / np.log(2)
        ent_l, ent_r = np.sum(- p_l * logp_l), np.sum(- p_r * logp_r)
        ent_after = (l * ent_l + r * ent_r) / (l + r)

        return ent - ent_after

    @staticmethod
    def split(X, y, idx, thresh):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(X.shape[0]):
            if X[i, idx] <= thresh:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return np.array(left_X), np.array(left_y), np.array(right_X), np.array(right_y)

    @staticmethod
    def test_split(X, idx, thresh):
        left_X, right_X = [], []
        for i in range(X.shape[0]):
            if X[i, idx] <= thresh:
                left_X.append(X[i])
            else:
                right_X.append(X[i])

        return np.array(left_X), np.array(right_X)

    def fit(self, X, y, allowed=None):
        if X.shape[0] == 0:
            self.label = 0
        elif self.max_depth == 1:
            if y.shape[0] > 0:
                self.label = stats.mode(y)[0][0]

            # print('label is', self.label)
        else:
            # Find best way to split
            n_features = X.shape[1]
            indexList = range(n_features)
            best_gain, best_idx, best_thresh = 0, 0, 0
            # if allowed is not None:
            #     n_features = np.array(allowed).shape[0]
            #     indexList = allowed

            for i in range(n_features):
                index = indexList[i]
                feature_data = np.transpose(X)[index]
                min_val, max_val = np.min(feature_data), np.max(feature_data)
                # stepsize, test_thresh = (max_val - min_val) / 12, min_val
                # for j in range(10):
                stepsize, test_thresh = (max_val - min_val) / 12, min_val
                for j in range(10):
                    test_thresh += stepsize
                    info_gain = self.information_gain(feature_data, y, test_thresh)
                    if info_gain > best_gain:
                        best_gain, best_idx, best_thresh = info_gain, index, test_thresh

            # Perform best split and make new nodes
            self.split_rule = [best_idx, best_thresh]
            # print(self.split_rule)
            left_X, left_y, right_X, right_y = self.split(X, y, best_idx, best_thresh)
            # print('left shape', left_X.shape, 'right shape', right_X.shape)
            self.left = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
            self.right = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
            # print('left side')
            self.left.fit(left_X, left_y, allowed=allowed)
            # print('right side')
            self.right.fit(right_X, right_y, allowed=allowed)
        return self

    def scalar_predict(self, X):
        if X.shape[0] == 0:
            return 0
        elif self.max_depth == 1:
            if self.label is None:
                self.label = 0

            # print('label is', self.label)
            return self.label
        else:
            if self.split_rule:
                idx, thresh = self.split_rule
                if X[idx] <= thresh:
                    # print(self.split_rule, 'going left')
                    return self.left.predict(X)
                else:
                    # print(self.split_rule, 'going right')
                    return self.right.predict(X)
            else:
                return 0


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=10, max_depth=3):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(max_depth=max_depth)
            for i in range(self.n)
        ]
        self.predict = np.vectorize(self.scalar_predict, signature='(n)->()')

    def fit(self, X, y, m):
        for tree in self.decision_trees:
            indices = np.random.randint(0, X.shape[0], size=(X.shape[0]))
            X_rand, Y_rand = X[indices], y[indices]
            allowed = np.random.permutation(range(X.shape[1]))[0:m]
            # print('allowed are', allowed)
            tree.fit(X_rand, Y_rand, allowed)
        return self

    def scalar_predict(self, X):
        # print('predicting')
        outputs = np.zeros(self.n)
        for i in range(self.n):
            outputs[i] = self.decision_trees[i].predict(X)

        # print('outputs are', outputs)
        return np.round(np.sum(outputs) / self.n)


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=10, m=10, max_depth=2):
        if params is None:
            params = {}
        self.n = n
        self.m = m
        self.bagged_tree = BaggedTrees(params=params, max_depth=max_depth, n=n)
        self.predict = np.vectorize(self.scalar_predict, signature='(n)->()')

    def fit(self, X, y):
        m = self.m
        return self.bagged_tree.fit(X, y, m)

    def scalar_predict(self, X):
        return self.bagged_tree.predict(X)


def results_to_csv(y_test, name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(name, index_label='Id')


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        features = ["pclass", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked"]
        # Load titanic data       
        path_train, path_test = 'datasets/titanic/titanic_training.csv', 'datasets/titanic/titanic_testing_data.csv'
        dataframe_train, dataframe_test = pd.read_csv(path_train), pd.read_csv(path_test)
        class_names = ["Died", "Survived"]

        # One-Hot Encoding Categorical Variables
        data_categorical, data_categorical_test = dataframe_train.select_dtypes(include=[object]), dataframe_test.select_dtypes(include=[object])
        label_enc = preprocessing.LabelEncoder()  # encoding categories as integers
        data_categorical = data_categorical.apply(label_enc.fit_transform)
        data_categorical_test = data_categorical_test.apply(label_enc.fit_transform)

        onehot_enc, onehot_enc_test = preprocessing.OneHotEncoder(), preprocessing.OneHotEncoder()  # now one-hot encoding
        onehot_enc.fit(data_categorical)
        data_onehot, data_onehot_test = onehot_enc.transform(data_categorical).toarray(), onehot_enc.transform(data_categorical_test).toarray()

        # Imputing and Normalizing Numerical Variables
        data_numerical, data_numerical_test = dataframe_train.select_dtypes('number'), dataframe_test.select_dtypes('number')
        mean_impute = SimpleImputer(missing_values=np.nan, strategy='mean')  # impute missing values as mean
        data_numerical, data_numerical_test = mean_impute.fit_transform(data_numerical), mean_impute.fit_transform(data_numerical_test)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # scale to (0, 1)
        data_numerical, data_numerical_test = scaler.fit_transform(data_numerical), scaler.fit_transform(data_numerical_test)

        # Rejoin data into 'cleaned' data set
        X_data, Y_data = np.concatenate([data_numerical[:, 1:], data_onehot], axis=1), data_numerical[:, 0]
        X_data_test = np.concatenate([data_numerical_test, data_onehot_test], axis=1)

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X_data, Y_data, X_data_test = data['training_data'], np.squeeze(data['training_labels']), data['test_data']
        class_names = ["Ham", "Spam"]

        # Normalize data
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # scale to (0, 1)
        # X_data = scaler.fit_transform(X_data)
        # X_data_test = scaler.fit_transform(X_data_test)

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    # print("Train size", X_data.shape)
    # print("Test size", X_data_test.shape)
    # print("\n\nPart 0: constant classifier")
    # print("Accuracy", 1 - np.sum(Y_data) / Y_data.size)
    ########################
    # Basic decision tree
    ########################
    print('==================================================')
    print("\n\nSimplified decision tree")

    # Validation
    #
    # dt = DecisionTree(max_depth=4)
    # n_total, n_val = X_data.shape[0], 100
    # X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=n_val, random_state=1234)
    #
    # dt.fit(X_train, Y_train)
    # predictions_train = dt.predict(X_train)
    # predictions_val = dt.predict(X_val)
    #
    # accuracy_train = 1 - np.sum(np.abs(predictions_train - Y_train)) / Y_train.shape[0]
    # print('Training Accuracy:', accuracy_train)
    # accuracy_val = 1 - np.sum(np.abs(predictions_val - Y_val)) / Y_val.shape[0]
    # print('Validation Accuracy:', accuracy_val)

    depths = [1, 2, 5, 10, 15, 20, 30, 40]
    accuracy_train_list = []
    accuracy_val_list = []
    n_total, n_val = X_data.shape[0], 1034
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=n_val, random_state=1234)

    for depth in depths:
        dt = DecisionTree(max_depth=depth)
        dt.fit(X_train, Y_train)
        predictions_train = dt.predict(X_train)
        predictions_val = dt.predict(X_val)
        accuracy_train = 1 - np.sum(np.abs(predictions_train - Y_train)) / Y_train.shape[0]
        print('Training Accuracy:', accuracy_train)
        accuracy_train_list.append(accuracy_train)
        accuracy_val = 1 - np.sum(np.abs(predictions_val - Y_val)) / Y_val.shape[0]
        print('Validation Accuracy:', accuracy_val)
        accuracy_val_list.append(accuracy_val)

    plt.plot(depths, accuracy_train_list, depths, accuracy_val_list)
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.ylim([.7, 1])
    plt.title("Spam Decision Tree Accuracies vs. maximum tree depth")
    plt.show()


    # # Test Prediction
    # dt = DecisionTree(max_depth=4)
    # dt.fit(X_data, Y_data)
    # predictions = dt.predict(X_data_test)
    # print("Predictions are", predictions)
    # print("Tree structure", dt.__repr__())
    # results_to_csv(predictions, 'spam_submission.csv')

    ########################
    # Random Forests
    ########################
    # print('==================================================')
    # print("\n\n Random Forest")
    # # Validation
    # rf = RandomForest(n=10, m=300, max_depth=4)
    # n_total, n_val = X_data.shape[0], 100
    # X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=n_val, random_state=1234)
    #
    # rf.fit(X_train, Y_train)
    # predictions_train = rf.predict(X_train)
    # predictions_val = rf.predict(X_val)
    # accuracy_train = 1 - np.sum(np.abs(predictions_train - Y_train)) / Y_train.shape[0]
    # print('Training Accuracy:', accuracy_train)
    # accuracy_val = 1 - np.sum(np.abs(predictions_val - Y_val)) / Y_val.shape[0]
    # print('Validation Accuracy:', accuracy_val)

    # Test
    # test_predictions = np.round(expit(X_test.dot(w_best)))
    # results_to_csv(test_predictions, 'spam_submission.csv')
