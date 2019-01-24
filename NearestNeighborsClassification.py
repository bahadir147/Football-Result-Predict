import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score
# Kütüphaneler.
import Accuracy
import DataSaveLoad
import GetData

n_neighbors = 5

# import some data to play with

Datas = GetData.GetData('Data\GlobalData.xlsx')

X_train = Datas[0]
y_train = Datas[1].values.ravel()
X_test = Datas[2]
y_test = Datas[3]


for weights in ['uniform']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier()

    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())

    #clf.fit(X_train, y_train)

    # Accuracy.evaluate(clf, X_test, y_test.values.ravel(),
    #                  X_train, y_train.values.ravel())
