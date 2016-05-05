#!/usr/bin/python

""" this example borrows heavily from the example
    shown on the sklearn documentation:

    http://scikit-learn.org/stable/modules/cross_validation.html

"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

iris = datasets.load_iris()
features = iris.data
labels = iris.target

###############################################################
### YOUR CODE HERE
###############################################################

### import the relevant code and make your train/test split
### name the output datasets features_train, features_test,
### labels_train, and labels_test

### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result
kf = KFold(len(features),  n_folds=10, shuffle=False, random_state=None)
for train_index, test_index in kf:
    features_train = features[train_index]
    features_test = features[test_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]

###############################################################

clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)

def submitAcc():
    return clf.score(features_test, labels_test)
