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
veriler = pd.read_csv('Data\T1.csv',error_bad_lines=False)

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


sonuc = pd.DataFrame(data = FullTimeResult , index = range(1377) ,columns=['FTR'])
sonuc2 = pd.DataFrame(data = HalfTimeResult , index = range(1377) ,columns=['HTR'])
sonuc3 = pd.DataFrame(data = EvSahibitakimlar , index = range(1377) ,columns=['HomeTeam'])
sonuc4 = pd.DataFrame(data = RakipTakimlar , index = range(1377) ,columns=['AwayTeam'])
sonuc5 = pd.DataFrame(data = sayisalVeriler , index = range(1377), columns=['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','PSH','PSD','PSA','WHH','WHD','WHA','VCH','VCD','VCA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5','BbAv<2.5','BbAH','BbAHh','BbMxAHH','BbAvAHH','BbMxAHA','BbAvAHA','PSCH','PSCD','PSCA'])

s=pd.concat([sonuc3,sonuc4,sonuc5],axis=1)



x_train,x_test,y_train,y_test = train_test_split(s,sonuc,test_size=0.33,random_state=0)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test);

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
y_pred=logreg.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

