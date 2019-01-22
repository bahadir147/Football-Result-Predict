# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Kütüphaneler.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer



#Veri Yükleme
veriler = pd.read_excel('Data\out.xlsx')

DataCount = len(veriler.index)

#Veri Ön İşleme

imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis=0)

sayisalVeriler = veriler.iloc[:,22:]

imputer=imputer.fit(sayisalVeriler)
sayisalVeriler=imputer.transform(sayisalVeriler)


EvSahibitakimlar = veriler.iloc[:,2:3]
RakipTakimlar = veriler.iloc[:,3:4]
FullTimeResult = veriler.iloc[:,6:7]
HalfTimeResult = veriler.iloc[:,9:10]

TestEvSahibi = ["Buyuksehyr"]
TestRakipTakim = ["Trabzonspor"]


labelencoder_X = LabelEncoder()

EvSahibitakimlar.values[:, 0] = labelencoder_X.fit_transform(EvSahibitakimlar.values[:, 0])
RakipTakimlar.values[:, 0] = labelencoder_X.fit_transform(RakipTakimlar.values[:, 0])
FullTimeResult.values[:, 0] = labelencoder_X.fit_transform(FullTimeResult.values[:, 0])
HalfTimeResult.values[:, 0] = labelencoder_X.fit_transform(HalfTimeResult.values[:, 0])



sonuc = pd.DataFrame(data = FullTimeResult , index = range(DataCount) ,columns=['FTR'])
sonuc2 = pd.DataFrame(data = HalfTimeResult , index = range(DataCount) ,columns=['HTR'])
sonuc3 = pd.DataFrame(data = EvSahibitakimlar , index = range(DataCount) ,columns=['HomeTeam'])
sonuc4 = pd.DataFrame(data = RakipTakimlar , index = range(DataCount) ,columns=['AwayTeam'])
sonuc5 = pd.DataFrame(data = sayisalVeriler , index = range(DataCount), columns=['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','PSH','PSD','PSA','WHH','WHD','WHA','VCH','VCD','VCA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5','BbAv<2.5','BbAH','BbAHh','BbMxAHH','BbAvAHH','BbMxAHA','BbAvAHA','PSCH','PSCD','PSCA'])

s=pd.concat([sonuc3,sonuc4,sonuc5],axis=1)



x_train,x_test,y_train,y_test = train_test_split(s,sonuc,test_size=0.33,random_state=0)

y_train = y_train.astype('int')


y_test = y_test.astype('int')
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test);


#Native Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())
print("Naive bayes:" ,gnb.score(X_test,y_test.values.ravel()))
 
#MLPClassifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(X_train, y_train.values.ravel())
print("MLPClassifier :" ,clf.score(X_test,y_test.values.ravel()))
#print(clf.score(X_test,y_test))
#labelencoder_X.transform(veriler.iloc[:,6:7])
#print(list(labelencoder_X.inverse_transform(clf.classes_)))
#print(clf.predict_proba(X_test))


#Lineer Regresion
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train.values.ravel())
print("Lineer Regresion :" ,reg.score(X_test,y_test.values.ravel()))


#LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train.values.ravel())
print("LinearDiscriminantAnalysis :" ,clf.score(X_test,y_test.values.ravel()))

#SGDClassifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X_train, y_train.values.ravel())
print("SGDClassifier :" ,clf.score(X_test,y_test.values.ravel()))

#NearestCentroid
from sklearn.neighbors.nearest_centroid import NearestCentroid

clf = NearestCentroid()
clf.fit(X_train, y_train.values.ravel())
print("NearestCentroid :" ,clf.score(X_test,y_test.values.ravel()))


#PLSRegression
from sklearn.cross_decomposition import PLSRegression
pls2 = PLSRegression(n_components=2)
pls2.fit(X_train, y_train.values.ravel())
print("PLSRegression :" ,pls2.score(X_test,y_test.values.ravel()))

#DecisionTreeClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train.values.ravel())
print("DecisionTreeClassifier :" ,clf.score(X_test,y_test.values.ravel()))

# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=500)
clf.fit(X_train, y_train.values.ravel())
print("ExtraTreesClassifier :" ,clf.score(X_test,y_test.values.ravel()))

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train, y_train.values.ravel())
print("RandomForestClassifier :" ,clf.score(X_test,y_test.values.ravel()))


# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=500)
clf.fit(X_train, y_train.values.ravel())
print("AdaBoostClassifier :" ,clf.score(X_test,y_test.values.ravel()))
