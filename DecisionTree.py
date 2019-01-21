# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Kütüphaneler.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer



#Veri Yükleme
veriler = pd.read_csv('Data\T1.csv')

DataCount = len(veriler.index)

#Veri Ön İşleme

imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis=0)

sayisalVeriler = veriler.iloc[:,22:].values

imputer=imputer.fit(sayisalVeriler)
sayisalVeriler=imputer.transform(sayisalVeriler)


EvSahibitakimlar = veriler.iloc[:,2:3]
RakipTakimlar = veriler.iloc[:,3:4]
FullTimeResult = veriler.iloc[:,6:7]
HalfTimeResult = veriler.iloc[:,9:10]


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



x_train,x_test,y_train,y_test = train_test_split(s,sonuc,test_size=0.20,random_state=0)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test);


#BEST PARAMS

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid ={
 'max_depth': [10,20,30],
 'max_features': [ 'sqrt', 'log2'],
 'min_samples_leaf': [4],
 'min_samples_split': [ 10],
 'n_estimators': [1200, 1400]}


c, r = y_train.shape
y_train = y_train.values.reshape(c,)

CV_rfc = GridSearchCV(n_jobs=-1 ,estimator=rfc, param_grid=param_grid, cv= 3,verbose=5)
CV_rfc.fit(X_train,y_train)
bestparams = CV_rfc.best_params_
print (CV_rfc.best_params_)


#EĞİTİM
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(**bestparams);

y_pred=clf.fit(X_train, y_train)

accuracy = y_pred.score(X_test,y_test)

print(accuracy)





