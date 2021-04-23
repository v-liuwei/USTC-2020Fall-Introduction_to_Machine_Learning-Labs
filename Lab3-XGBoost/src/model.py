import numpy as np
from typing import Optional, Tuple, Union, Callable
from matplotlib import pyplot as plt
from metric import mean_squared_error as mse, error_rate as err

eps = 1e-10


class XGBoost(object):
    def __init__(self,
                 n_estimators: int,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 objective: Union[str,
                                  Callable[[np.ndarray, np.ndarray],
                                           Tuple[float,
                                                 float]]] = 'binary:logistic',
                 gamma: float = 0,
                 reg_lambda: float = 1.0,
                 subsample: float = 1.0,
                 colsample: float = 1.0,
                 random_state: Optional[int] = None):
        """
        n_estimators: 子树棵树，迭代次数
        max_depth: 每颗子树的最大允许深度
        learning_rate: 学习率，与shrinkage技术有关，详见原论文
        objective: 损失函数
        gamma: 控制叶结点个数的正则化系数
        reg_lambda: 二次正则化系数
        subsample: 构建每颗树时使用的样本比例，
        colsample: 构建每颗树时寻找分裂点时考虑的特征比例，参考random forest中的技术
        random_state: 随机种子
        """
        self.max_iter = n_estimators
        self.max_depth = max_depth
        self.lr = learning_rate
        self.obj = self.__getObj(objective)
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.seed = random_state

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: Optional[int] = None):
        """
        X,y: 拟合数据(X,y)
        eval_set: 评估训练过程中模型的泛化能力
        early_stopping_rounds：如果使用了eval_set, early_stopping_rounds用于控制模型的早停，
            当模型在eval_set上的损失连续 `early_stopping_rounds` 次迭代都没有降低，则停止迭代，
            并把损失最低的那次迭代次数记为最佳迭代次数
        """
        assert X.shape[0] == y.size
        self.train_data = (X, y)
        self.val_data = eval_set
        self.err = {'train': [], 'val': []}
        self.best_iter = self.max_iter
        self.trees = []
        BaseTree.set_data(X, y)
        cur_y = np.zeros(y.size)
        if self.val_data is not None:
            cur_val_y = np.zeros(self.val_data[1].size)
            if early_stopping_rounds:
                min_val_loss = np.inf
                non_dec_rounds = 0
        for i in range(self.max_iter):
            subtree = BaseTree(self.obj, self.max_depth, self.gamma,
                               self.reg_lambda, self.colsample, self.subsample,
                               self.seed)
            subtree.fit(cur_y)
            self.trees.append(subtree)

            cur_y += self.lr * subtree.predict(X)
            self.__update_err(y, cur_y, which='train')
            if self.val_data is not None:
                cur_val_y += self.lr * subtree.predict(self.val_data[0])
                val_loss = self.__update_err(self.val_data[1],
                                             cur_val_y,
                                             which='val')
                if early_stopping_rounds:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        non_dec_rounds = 0
                        self.best_iter = i + 1
                    else:
                        non_dec_rounds += 1
                        if non_dec_rounds >= early_stopping_rounds:
                            break
        return self

    def predict(self, X: np.ndarray, raw=False):
        """
        X: 预测X的输出
        raw: 当进行分类任务时，raw置为True将返回未经过logistic函数转换的原始预测结果
        """
        pred = np.zeros(X.shape[0])
        for subtree in self.trees[:self.
                                  best_iter] if self.best_iter else self.trees:
            pred += self.lr * subtree.predict(X)
        if self.task_type == 'clf' and not raw:
            pred = 1. / (1. + np.exp(-pred))
            pred = pred >= .5
        return pred

    def plot_learning_curve(self):
        fig, ax = plt.subplots()
        ax.set_title('Learning curve with lr={}'.format(self.lr))
        ax.set_xlabel('iter')
        if self.task_type == 'clf':
            ax.set_ylabel('error rate')
        else:
            ax.set_ylabel('mean squared error')
        ax.plot(np.arange(1,
                          len(self.err['train']) + 1),
                self.err['train'],
                label='training error')
        if self.err['val']:
            ax.plot(np.arange(1,
                              len(self.err['val']) + 1),
                    self.err['val'],
                    label='validation error')
        ax.legend()
        plt.show()

    def __getObj(self, objective):
        if type(objective) == str:
            _objective = objective.lower()
            if _objective == 'binary:logistic':
                self.task_type = 'clf'

                def obj(ytrue, ypred):
                    ypred = 1. / (1. + np.exp(-ypred))
                    grad = ypred - ytrue
                    hess = ypred * (1 - ypred)
                    return grad, hess
            elif _objective == 'reg:squarederror':
                self.task_type = 'reg'

                def obj(ytrue, ypred):
                    return 2 * (ypred - ytrue), 2
            else:
                raise ValueError(f'not supported objective: `{objective}`')
            return obj
        elif callable(objective):
            self.task_type = 'reg'
            return objective
        else:
            raise TypeError("`objective` should be str or callable")

    def __update_err(self, ytrue, ypred, which='train'):
        if self.task_type == 'clf':
            ypred = 1. / (1. + np.exp(-ypred))
            ypred = (ypred >= .5)
            loss = err(ytrue, ypred)
            self.err[which].append(loss)
        else:
            loss = mse(ytrue, ypred)
            self.err[which].append(loss)
        return loss


class TreeNode(object):
    def __init__(self, indices):
        self.indices = indices
        self.isleaf = False
        self.split_k = None  # 分裂点选取的特征
        self.split_v = None  # 分裂点选取的特征的值
        self.score = None
        self.left = None
        self.right = None


class BaseTree(object):
    __X = None
    __y = None

    @staticmethod
    def set_data(X: np.ndarray, y: np.ndarray):
        assert (X.shape[0] == y.size)
        if BaseTree.__X is None:
            BaseTree.__X = X
            BaseTree.__y = y

    def __init__(self, objective, max_depth, gamma, reg_lambda, subsample,
                 colsample, seed):
        self.obj = objective
        self.max_depth = max_depth
        self.delta = gamma
        self.c = reg_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.seed = seed

    def fit(self, current_y: np.ndarray):
        self.__g, self.__h = np.array([
            self.obj(y_true, y_pred)
            for y_true, y_pred in zip(BaseTree.__y, current_y)
        ]).T
        np.random.seed(self.seed)
        m, p = BaseTree.__X.shape
        self.row_samples = np.random.choice(m,
                                            round(self.subsample * m),
                                            replace=False)
        self.col_samples = np.random.choice(p,
                                            round(self.colsample * p),
                                            replace=False)
        self.tree = self.__construct(TreeNode(self.row_samples), depth=0)
        return self

    def predict(self, X: np.ndarray):
        pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            node = self.tree
            while not node.isleaf:
                if x[node.split_k] <= node.split_v:
                    node = node.left
                else:
                    node = node.right
            pred[i] = node.score
        return pred

    def __split(self, indices):
        if len(indices) < 2:
            return False, None
        gain = 0
        _k = _i = None
        G, H = np.sum(self.__g[indices]), np.sum(self.__h[indices])
        for k in self.col_samples:
            Gleft, Hleft = 0, 0
            Gright, Hright = G, H

            sorted_indices = sorted(indices, key=lambda i: BaseTree.__X[i, k])
            i = 0
            while i < len(sorted_indices) - 1:
                idx = sorted_indices[i]
                cur_value = BaseTree.__X[idx, k]
                Gleft += self.__g[idx]
                Hleft += self.__h[idx]
                while i + 1 < len(sorted_indices) - 1 and BaseTree.__X[
                        sorted_indices[i + 1], k] == cur_value:
                    i = i + 1
                    idx = sorted_indices[i]
                    Gleft += self.__g[idx]
                    Hleft += self.__h[idx]
                Gright, Hright = G - Gleft, H - Hleft
                cur_gain = Gleft**2 / (Hleft + self.c + eps) + Gright**2 / (
                    Hright + self.c + eps) - G**2 / (H + self.c + eps)
                if cur_gain > gain:
                    gain, _k, _i = cur_gain, k, i
                i = i + 1
        if gain > self.delta:
            sorted_indices = sorted(indices, key=lambda i: BaseTree.__X[i, _k])
            return True, (_k, BaseTree.__X[sorted_indices[_i], _k],
                          sorted_indices[:_i + 1], sorted_indices[_i + 1:])
        else:
            return False, None

    def __construct(self, node: TreeNode, depth):
        split = True
        if depth >= self.max_depth:
            split = False
        if split:
            split, split_res = self.__split(node.indices)
        if split:
            k, v, left_indices, right_indices = split_res
            node.split_k, node.split_v = k, v
            node.left = self.__construct(TreeNode(left_indices), depth + 1)
            node.right = self.__construct(TreeNode(right_indices), depth + 1)
        else:
            node.isleaf = True
            G, H = np.sum(self.__g[node.indices]), np.sum(
                self.__h[node.indices])
            node.score = -G / (H + self.c + eps)
        return node
