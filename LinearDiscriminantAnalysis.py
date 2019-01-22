# Kütüphaneler.
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#!/usr/bin/python
import GetData

Datas = GetData.GetData()

X_train=Datas[0]
y_train=Datas[1]
X_test=Datas[2]
y_test=Datas[3]


clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
 
print('Train Score: ', clf.score(X_train, y_train))
print('Test Score: ', clf.score(X_test, y_test))