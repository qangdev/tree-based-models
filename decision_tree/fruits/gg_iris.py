
from sklearn import tree
# [gram, texture]
# texture:
#     - 0 = Bumpy
#     - 1 = Smooth
# label:
#     - 0 = Apple
#     - 1 = Orange

# features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labels = [0, 0, 1, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features, labels)
# print(clf.predict([[150, 0]]))

import numpy as np
from sklearn.datasets import load_iris

# S T E P  0: import/Load dataset
dataset_iris = load_iris()
# print(dataset_iris.__dir__())
# print(dataset_iris.data)
# print(dataset_iris.target)


# S T E P  1: partition dataset into `test` and `train` sub-datasets
indexes = [0, 50, 100]  # which indexes to be removed from dataset
# make `train`
train_data = np.delete(dataset_iris.data, indexes, axis=0)  # axis = 0 indicating row (Needed because of 2d array)
train_target = np.delete(dataset_iris.target, indexes)
# print(train_data[:10])
# print(train_target[:10])
# make `test`
test_data = dataset_iris.data[indexes]
test_target = dataset_iris.target[indexes]
# print(test_data[:10])
# print(test_target[:10])


# S T E P  2: define a `classifier`
from sklearn import tree
clf = tree.DecisionTreeClassifier()


# S T E P  3: train the `classifier`
clf.fit(train_data, train_target)


# S T E P  4: make one prediction
results = clf.predict(test_data)


# S T E P  4.1: make many predictions [Optional]


# S T E P  5: evaluate the result
print(list(results), "<>", list(test_target))
assert list(results) == list(test_target)

print("[d o n e]")

