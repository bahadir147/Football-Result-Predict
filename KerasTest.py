# Kütüphaneler.
from keras.layers import Dense
from keras.models import Sequential
import keras
import numpy as np
import sklearn


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

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)
y_train=enc.transform(y_train)

classifier = Sequential()

classifier.add(Dense(3, init='uniform', activation='relu', input_dim=39))
classifier.add(Dense(3, init='uniform', activation='relu'))
classifier.add(Dense(3, init='uniform', activation='sigmoid'))
classifier.add(Dense(3, init='uniform', activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=70)

y_pred = classifier.predict(X_test)
print(y_pred)

print("-------------------------")

print(y_test)

#Accuracy.evaluate(classifier, X_test, y_test, X_train, y_train)
