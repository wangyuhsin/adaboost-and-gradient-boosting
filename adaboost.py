import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def parse_spambase_data(filename):
    """
    Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    p = Path(filename)
    data = p.read_text().split("\n")
    data.remove("")

    X = np.zeros((len(data), len(data[0].split(",")) - 1))
    Y = np.zeros((len(data),))

    for i in range(len(data)):
        row = data[i].split(",")
        for j in range(len(row)):
            if j != len(row) - 1:
                X[i][j] = np.float64(row[j])
            else:
                Y[i] = -1 if int(row[j]) == 0 else 1

    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """
    Given an numpy matrix X, a array y and num_iter return trees and weights

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
             
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    d = np.ones(N) / N

    w = d
    for m in range(num_iter):
        h = DecisionTreeClassifier(max_depth=1, random_state=0)
        h.fit(X, y, sample_weight=w)
        y_pred = h.predict(X)

        trees.append(h)

        err = np.sum(w * (y_pred != y)) / np.sum(w)
        alpha = np.log((1 - err) / err)
        w *= np.exp(alpha * (y_pred != y))
        trees_weights.append(alpha)

    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """
    Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ = X.shape
    y = np.zeros(N)

    preds = []
    for i in range(len(trees)):
        y_pred = trees[i].predict(X)
        preds.append(trees_weights[i] * y_pred)
    y = np.sign(np.sum(preds, axis=0))

    return y
