import numpy as np
from typing import Optional, Tuple
from typing_extensions import Literal
from matplotlib import pyplot as plt


class LogisticRegression(object):
    def __init__(self,
                 learning_rate: float,
                 max_iter: int,
                 fit_bias: Optional[bool] = True,
                 optimizer: Literal['SGD', 'GD', 'mbSGD', 'Newton', None] = None,
                 batch_size: Optional[int] = None,
                 seed: Optional[int] = None):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.bias = fit_bias
        self.optim = optimizer if optimizer else 'SGD'
        self.BATCH_SIZE = batch_size
        self.seed = seed

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.train_data = (X, y)
        self.val_data = val_data
        np.random.seed(self.seed)
        X = self.__transfrom(X)
        self.w = np.random.normal(scale=0.1, size=X.shape[1])
        self.err = {'train': [], 'val': []}
        if self.optim == 'SGD':
            _batch_size = 1
            _method = 'GD'
        if self.optim == 'GD':
            _batch_size = X.shape[0]
            _method = 'GD'
        if self.optim == 'mbSGD':
            if self.BATCH_SIZE:
                _batch_size = self.BATCH_SIZE
                _method = 'GD'
            else:
                raise ValueError(
                    'Use a `mbSGD` optimizer but parameter `batch_size` is not specified'
                )
        if self.optim == 'Newton':
            _batch_size = X.shape[0]
            _method = 'Newton'
        for i in range(self.max_iter):
            self.__batch_BP(X, y, _batch_size, _method)
            self.__update_err()

    def predict(self, X: np.ndarray):
        X = self.__transfrom(X)
        return np.array([(1 / (1 + np.exp(-np.dot(self.w, x))) >= 0.5)
                         for x in X])

    def predict_proba(self, X: np.ndarray):
        X = self.__transfrom(X)
        return np.array([1 / (1 + np.exp(-np.dot(self.w, x))) for x in X])

    def score(self,
              X: np.ndarray,
              target: np.ndarray,
              metric: Literal['err', 'acc', 'mse', 'rmse', 'f1'] = 'acc'):
        assert (X.shape[0] == target.size)
        if metric == 'acc' or 'err':
            y_pred = self.predict(X)
            acc = np.sum(y_pred == target) / target.size
            return acc if metric == 'acc' else 1 - acc
        if metric == 'mse' or 'rmse':
            y_pred_score = self.predict_proba(X)
            mse = np.sum((y_pred_score - target)**2) / target.size
            return mse if metric == 'mse' else np.sqrt(mse)
        if metric == 'f1':
            y_pred = self.predict(X)
            TP = np.sum(np.logical_and(y_pred == 1, target == 1))
            prec = TP / np.sum(y_pred == 1)
            recall = TP / np.sum(target == 1)
            return 2 * prec * recall / (prec + recall)

    def plot_boundary(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      type: Literal['scatter'] = 'scatter'):
        x1, x2 = X[:, 0], X[:, 1]
        x1_lim = np.array([np.min(x1), np.max(x1)
                           ]) + np.array([-1, 1]) * .05 * np.ptp(x1)
        x2_lim = np.array([np.min(x2), np.max(x2)
                           ]) + np.array([-1, 1]) * .05 * np.ptp(x2)

        fig, ax = plt.subplots()
        ax.set_xlim(x1_lim[0], x1_lim[1])
        ax.set_ylim(x2_lim[0], x2_lim[1])
        acc = self.score(X, y, 'acc')
        ax.set_title("Boundary of Logistic Regression\naccuracy={}".format(round(acc, 3)))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        x1_sample = np.linspace(np.floor(x1_lim[0]),
                                np.ceil(x1_lim[1]),
                                num=100)
        if self.bias:
            x2_sample = -(self.w[0] + self.w[1] * x1_sample) / (self.w[2] +
                                                                1e-10)
        else:
            x2_sample = -(self.w[0] * x1_sample) / (self.w[1] + 1e-10)
        ax.plot(x1_sample, x2_sample, label='boundary')
        ax.scatter(x1[y == 1], x2[y == 1], c='green', label='positive')
        ax.scatter(x1[y == 0], x2[y == 0], c='red', label='negative')
        ax.legend(loc='upper left')
        plt.show()

    def plot_learning_curve(self):
        fig, ax = plt.subplots()
        ax.set_title('Learning curve with lr={}'.format(self.lr))
        ax.set_xlabel('epoch')
        ax.set_ylabel('error rate')
        ax.plot(np.arange(1,
                          len(self.err['train']) + 1),
                self.err['train'],
                label='training error')
        if self.err['val']:
            ax.plot(np.arange(1,
                              len(self.err['val']) + 1),
                    self.err['val'],
                    label='validating error')
        ax.legend()
        plt.show()

    def __transfrom(self, X: np.ndarray):
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        return X

    def __update_err(self):
        self.err['train'].append(
            self.score(self.train_data[0], self.train_data[1], 'err'))
        if self.val_data:
            self.err['val'].append(
                self.score(self.val_data[0], self.val_data[1], 'err'))

    def __data_loader(self, X, y, batch_size):
        batch_num = X.shape[0] // batch_size
        for b in range(batch_num):
            yield X[b * batch_size:(b + 1) *
                    batch_size], y[b * batch_size:(b + 1) * batch_size]
        if X.shape[0] % batch_size != 0:
            yield X[batch_num * batch_size:], y[batch_num * batch_size:]

    def __batch_BP(self,
                   X,
                   y,
                   batch_size,
                   method: Literal['GD', 'Newton'] = 'GD'):
        for batch_X, batch_y in self.__data_loader(X, y, batch_size):
            grad = np.zeros(self.w.size)
            hess = np.mat(np.zeros((self.w.size, self.w.size)))
            for _x, _y in zip(batch_X, batch_y):
                p1 = 1 / (1 + np.exp(-np.dot(self.w, _x)))
                grad += (p1 - _y) * _x
                if method == 'Newton':
                    hess += p1 * (1 - p1) * np.dot(np.mat(_x).T, np.mat(_x))
            if method == 'GD':
                self.w -= self.lr * grad / batch_y.size
            if method == 'Newton':
                self.w -= self.lr * np.dot(np.linalg.pinv(hess), grad).A[0]
