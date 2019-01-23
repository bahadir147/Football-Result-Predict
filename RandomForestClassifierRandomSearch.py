# Kütüphaneler.
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=10, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=3, min_samples_split=3,
                                 min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                                 oob_score=False, random_state=None, verbose=0,
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

        # Number of trees in random forest
        n_estimators = [int(x)
                        for x in np.linspace(start=10, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Set the parameters by cross-validation

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                       n_iter=250, cv=4, verbose=5, random_state=0, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(X_train, y_train.values.ravel())
        print("Best Score: ", rf_random.best_score_)
        print("Best Estimatör: ", rf_random.best_estimator_)
