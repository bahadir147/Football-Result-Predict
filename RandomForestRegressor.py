# Kütüphaneler.
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#!/usr/bin/python
import GetData

Datas = GetData.GetData()

X_train=Datas[0]
y_train=Datas[1]
X_test=Datas[2]
y_test=Datas[3]

    

clf = RandomForestRegressor(n_estimators=20)

clf.fit(X_train,y_train.values.ravel())

print("Training Accuracy = ", clf.score(X_train, y_train))
print("Test Accuracy = ", clf.score(X_test, y_test))


#
if __name__ == '__main__':
    # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators': [500, 700, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3,4]}

    # clf = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1)
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, 
                   n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    print (clf.best_estimator_)