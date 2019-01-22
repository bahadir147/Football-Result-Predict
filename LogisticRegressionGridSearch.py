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


#BEST PARAMS

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# Build a classification task using 3 informative features

bestparams={}
if __name__ == '__main__':
    logreg=LogisticRegression()

    grid={
            "C":np.logspace(-3,3,7,1,6), 
            "penalty":["l1","l2"],
            'fit_intercept':[True, False],
            'intercept_scaling':[0.1,0.2,0.25,1,2],
            "max_iter":[300,400,500,600,1000,2000],
            "tol":[1e-4,1e-5,1e-3]
            }
    
    logreg=LogisticRegression()
    logreg_cv=GridSearchCV(logreg,grid,cv=5,verbose=10,n_jobs=-1)
    logreg_cv.fit(x_train,y_train.values.ravel())
    
    
#    c, r = y_train.shape
#    y_train = y_train.values.reshape(c,)
#
#    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,verbose=20,n_jobs=-1)
#    CV_rfc.fit(X_train,y_train)
    bestparams = logreg_cv.best_params_
    print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
    print("accuracy :",logreg_cv.best_score_)

#EĞİTİM
    
logreg2 = LogisticRegression(C=1,penalty="l2")

y_pred = logreg2.fit(x_train,y_train.values.ravel())

print("Accuracyy: ",logreg2.score(x_test,y_test))







