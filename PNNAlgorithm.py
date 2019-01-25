import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms, environment
import GetData
import DataSaveLoad
import Accuracy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

Datas = GetData.GetData('Data\GlobalData.xlsx')

x_train = Datas[0]
y_train = Datas[1].values
x_test = Datas[2]
y_test = Datas[3].values


pnn = algorithms.PNN(std=0.5, verbose=False, batch_size=35)
pnn.train(x_train, y_train)

y_predicted = pnn.predict(x_test)
print(metrics.accuracy_score(y_test, y_predicted))

Accuracy.evaluate(pnn, x_test, y_test.ravel(),
                  x_train, y_train.ravel())
