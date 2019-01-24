# Kütüphaneler.
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import keras
import numpy as np
import sklearn
from keras import optimizers
from keras.layers import *
from keras.models import *


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

##enc = OneHotEncoder( n_values=10)
# enc.fit(y_train)
#y_train = enc.transform(y_train).toarray()


X_train = X_train.reshape(6766, 39)
X_test = X_test.reshape(3333, 39)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


nb_classes = 10
uniques, ids = np.unique(y_train, return_inverse=True)
coded_array = np_utils.to_categorical(ids, 20)

#y_test = np_utils.to_categorical(y_test, nb_classes)

init = 'normal'

classifier = Sequential()

classifier.add(Dense(20, input_shape=(39,)))
# An "activation" is just a non-linear function applied to the output
classifier.add(Activation('relu'))
# Dropout helps protect the model from memorizing or "overfitting" the training data
classifier.add(Dropout(0.2))
classifier.add(Dense(512, init=init))
classifier.add(Dense(512, init=init))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(20, init=init))
classifier.add(Activation('softmax'))  # This special "softmax" a

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, coded_array, epochs=10, batch_size=20,verbose=1)

y_pred = classifier.predict(X_test)

y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]
# print(y_classes)

print(uniques[y_pred.argmax(1)])

print("-------------------------")


print(y_test)


#np.savetxt("y_Test(Gercek).csv", y_test, delimiter=",")
#np.savetxt("y_Pred(Predicted).csv", y_pred, delimiter=",")
#Accuracy.evaluate(classifier, X_test, y_test, X_train, y_train)


print("Accuracy Score: ")
matrix = confusion_matrix(y_test, y_classes)
print(matrix)


print(accuracy_score(y_test, y_classes))
