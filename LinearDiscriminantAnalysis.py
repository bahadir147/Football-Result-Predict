# Kütüphaneler.
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#!/usr/bin/python
import GetData
import DataSaveLoad
import Accuracy


TunnigData = True


Datas = GetData.GetData('Data\GlobalData.xlsx')

X_train = Datas[0]
y_train = Datas[1]
X_test = Datas[2]
y_test = Datas[3]



clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
 
Accuracy.evaluate(clf, X_test, y_test.values.ravel(),
                      X_train, y_train.values.ravel())