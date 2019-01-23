# Kütüphaneler.
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#!/usr/bin/python
import GetData
import DataSaveLoad


TunnigData = False


Datas = GetData.GetData('Data\FTOTAL(1234) Extra.xls')

X_train = Datas[0]
y_train = Datas[1]
X_test = Datas[2]
y_test = Datas[3]


# print(X_train)
# print(y_train)

if TunnigData == False:

    clf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                 max_depth=12, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=3, min_samples_split=3,
                                 min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                                 oob_score=False, random_state=0, verbose=0,
                                 warm_start=False)

    clf.fit(X_train, y_train.values.ravel())

    print("Training Accuracy = ", clf.score(X_train, y_train.values.ravel()))
    print("Test Accuracy = ", clf.score(X_test, y_test.values.ravel()))

    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Matrix:")
    print(conf_mat)

    # SAVE DATA
    dataFileName = "RandomForestClassifier"
    DataSaveLoad.SaveData(clf, dataFileName)


#
if TunnigData == True:
    if __name__ == '__main__':
        # Set the parameters by cross-validation
        tuned_parameters = {'bootstrap': [True, False],
                            'max_depth': [10, 20],
                            'max_features': ['auto', 'sqrt'],
                            'min_samples_leaf': [2, 4],
                            'min_samples_split': [2, 5, 10],
                            'n_estimators': [200, 600, 1000],
                            'min_weight_fraction_leaf': [0.0, 0.3, 0.5],
                            'min_impurity_decrease': [0.0, 0.5, 1.0]
                            }

        clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, cv=4,
                           n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train.values.ravel())
        print("Best Score: ", clf.best_score_)
        print("Best Estimatör: ", clf.best_estimator_)
