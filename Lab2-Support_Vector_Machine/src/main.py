import numpy as np
from model import SVMClassifier

# load data
x_train = np.load('./data/s-svm/train_data.npy')
y_train = np.load('./data/s-svm/train_target.npy')
x_test = np.load('./data/s-svm/test_data.npy')
y_test = np.load('./data/s-svm/test_target.npy')

# create an svm model and fit it
svm = SVMClassifier(learning_rate=0.001, max_iter=5000, C=1, optimizer='GD', seed=0)
svm.fit(x_train, y_train, val_data=(x_test, y_test))

# predict and calculate acc
train_acc = svm.score(x_train, y_train, metric='acc')
test_acc = svm.score(x_test, y_test, metric='acc')
print("train acc = {0}".format(train_acc))
print("test acc = {0}".format(test_acc))

# plot learning curve and decision boundary
svm.plot_learning_curve()
svm.plot_boundary(x_train, y_train, sv=True)
svm.plot_boundary(x_test, y_test, sv=False)
