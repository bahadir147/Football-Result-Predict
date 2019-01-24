# Kütüphaneler.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#!/usr/bin/python
import GetData
import DataSaveLoad
import Accuracy


TunnigData = False


Datas = GetData.GetData('Data\GlobalData.xlsx')

X_train = Datas[0]
y_train = Datas[1]
X_test = Datas[2]
y_test = Datas[3]


# print(X_train)
# print(y_train)

if TunnigData == False:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=5, max_features=2,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

    clf.fit(X_train, y_train.values.ravel())

    #print("Training Accuracy = ", clf.score(X_train, y_train.values.ravel()))
    #print("Test Accuracy = ", clf.score(X_test, y_test.values.ravel()))

    Accuracy.evaluate(clf, X_test, y_test.values.ravel(),
                      X_train, y_train.values.ravel())

    # SAVE DATA
    dataFileName = "RandomForestClassifier"
    DataSaveLoad.SaveData(clf, dataFileName)


#
if TunnigData == True:
    if __name__ == '__main__':
        # Set the parameters by cross-validation
        tuned_parameters = {
            'bootstrap': [True, False],
            'max_depth': [5, 10,20],
            'max_features': [2, 3],
            # 'min_samples_leaf': [3, 4, 5],
            # 'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300]
        }

        clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, cv=4,
                           n_jobs=-1, verbose=10)
        clf.fit(X_train, y_train.values.ravel())
        print("Best Score: ", clf.best_score_)
        print("Best Estimatör: ", clf.best_estimator_)
