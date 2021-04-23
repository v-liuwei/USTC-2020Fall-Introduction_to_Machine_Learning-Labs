import numpy as np
from model import XGBoost
from metric import accuracy as acc
import time

# load data
x_train_ = np.load('./data/train_data.npy')
y_train_ = np.load('./data/train_target.npy')
x_test = np.load('./data/test_data.npy')
y_test = np.load('./data/test_target.npy')

# split train data into train and eval data
m, p = x_train_.shape
np.random.seed(123)
indices = np.arange(m)
np.random.shuffle(indices)
x_train = x_train_[indices[:round(0.8 * m)]]
y_train = y_train_[indices[:round(0.8 * m)]]
x_eval = x_train_[indices[round(0.8 * m):]]
y_eval = y_train_[indices[round(0.8 * m):]]


def default_params():
    # create an xgboost model and fit it
    xgb = XGBoost(
        n_estimators=100,
        random_state=123)
    xgb.fit(x_train, y_train, eval_set=(x_eval, y_eval))

    # predict and calculate acc
    ypred_train = xgb.predict(x_train)
    ypred_eval = xgb.predict(x_eval)
    ypred_test = xgb.predict(x_test)
    print("train acc = {0}".format(acc(y_train, ypred_train)))
    print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
    print("test acc = {0}".format(acc(y_test, ypred_test)))

    # plot learning curve to tune parameter
    xgb.plot_learning_curve()


def early_stop():
    # create an xgboost model and fit it
    xgb = XGBoost(
        n_estimators=100,
        random_state=123)
    xgb.fit(x_train, y_train, eval_set=(x_eval, y_eval), early_stopping_rounds=20)
    print('best iter: {}'.format(xgb.best_iter))

    # predict and calculate acc
    ypred_train = xgb.predict(x_train)
    ypred_eval = xgb.predict(x_eval)
    ypred_test = xgb.predict(x_test)
    print("train acc = {0}".format(acc(y_train, ypred_train)))
    print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
    print("test acc = {0}".format(acc(y_test, ypred_test)))

    # plot learning curve to tune parameter
    xgb.plot_learning_curve()


def tuned_params():
    # create an xgboost model and fit it
    xgb = XGBoost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        gamma=0,
        reg_lambda=3,
        subsample=1,
        colsample=1,
        random_state=123)
    xgb.fit(x_train, y_train, eval_set=(x_eval, y_eval), early_stopping_rounds=20)
    print('best iter: {}'.format(xgb.best_iter))

    # predict and calculate acc
    ypred_train = xgb.predict(x_train)
    ypred_eval = xgb.predict(x_eval)
    ypred_test = xgb.predict(x_test)
    print("train acc = {0}".format(acc(y_train, ypred_train)))
    print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
    print("test acc = {0}".format(acc(y_test, ypred_test)))

    # plot learning curve to tune parameter
    # xgb.plot_learning_curve()


if __name__ == "__main__":
    s = time.time()

    # use default parameters without tuning
    # default_params()

    # use early stop
    # early_stop()
    # tuning parameters
    tuned_params()
    print(time.time() - s)
