# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Kütüphaneler.
import numpy as np
import matplotlib.pyplot as plt
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



x_train,x_test,y_train,y_test = train_test_split(s,sonuc,test_size=0.25,random_state=0)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test);


#BEST PARAMS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
# Build a classification task using 3 informative features


# build a classifier
bestparams={}
if __name__ == '__main__':
    clf = RandomForestClassifier(n_estimators=20,random_state=0)


# specify parameters and distributions to sample from
    param_dist = {
              'bootstrap': [True, False],
              "n_estimators":sp_randint(10,3000), 
              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              "max_features": sp_randint(1, 41),
              "min_samples_split": sp_randint(2, 41),
              'min_samples_leaf': sp_randint(2, 30),
              "criterion": ["gini", "entropy"]
              }

# run randomized search
    n_iter_search = 10
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=4, verbose =10, n_jobs=-1)

    random_search.fit(X_train, y_train.values.ravel())
    bestparams = random_search.best_params_
    print(random_search.best_score_)

#EĞİTİM
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(**bestparams);

y_pred=clf.fit(X_train, y_train.values.ravel())

accuracy = y_pred.score(X_test,y_test)

print(accuracy)





