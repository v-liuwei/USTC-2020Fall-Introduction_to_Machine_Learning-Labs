import numpy as np
from model import LogisticRegression


# load data
x_train = np.load('./data/LR/train_data.npy')[:, 1:]
y_train = np.load('./data/LR/train_target.npy')
x_test = np.load('./data/LR/test_data.npy')[:, 1:]
y_test = np.load('./data/LR/test_target.npy')

# create an LR model and fit it
lr = LogisticRegression(learning_rate=1, max_iter=10, fit_bias=True, optimizer='Newton', seed=0)
lr.fit(x_train, y_train, val_data=(x_test, y_test))

# predict and calculate acc
train_acc = lr.score(x_train, y_train, metric='acc')
test_acc = lr.score(x_test, y_test, metric='acc')
print("train acc = {0}".format(train_acc))
print("test acc = {0}".format(test_acc))

# plot learning curve and decision boundary
lr.plot_learning_curve()
lr.plot_boundary(x_train, y_train)
lr.plot_boundary(x_test, y_test)
