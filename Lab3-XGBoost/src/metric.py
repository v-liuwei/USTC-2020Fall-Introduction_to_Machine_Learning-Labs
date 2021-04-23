import numpy as np


def accuracy(ytrue: np.ndarray, ypred: np.ndarray):
    return np.sum(ytrue == ypred) / ytrue.size


def error_rate(ytrue: np.ndarray, ypred: np.ndarray):
    return np.sum(ytrue != ypred) / ytrue.size


def mean_squared_error(ytrue: np.ndarray, ypred: np.ndarray):
    return np.sum((ytrue - ypred) ** 2) / ytrue.size


def rooted_mean_squared_error(ytrue: np.ndarray, ypred: np.ndarray):
    return np.sqrt(np.sum((ytrue - ypred) ** 2) / ytrue.size)


def f1_score(ytrue: np.ndarray, ypred: np.ndarray):
    tp = np.sum(np.logical_and(ytrue == 1, ypred == 1))
    recall = tp / np.sum(ytrue == 1)
    prec = tp / np.sum(ypred == 1)
    return 2 * prec * recall / (prec + recall)
