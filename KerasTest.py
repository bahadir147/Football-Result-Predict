# Kütüphaneler.
import Accuracy
import DataSaveLoad
import GetData
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import keras
import numpy as np
import sklearn
from keras import optimizers
from keras.layers import *
from keras.models import *
import os
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#!/usr/bin/python


TunnigData = False


Datas = GetData.GetData('Data\GlobalData.xlsx')

X_train = Datas[0]
y_train = Datas[1]
X_test = Datas[2]
y_test = Datas[3]

##enc = OneHotEncoder( n_values=10)
# enc.fit(y_train)
#y_train = enc.transform(y_train).toarray()


#X_train = X_train.reshape(6766, 39)
#X_test = X_test.reshape(3333, 39)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train -1
nb_classes = 3
uniques, ids = np.unique(y_train, return_inverse=True)
y_train = np_utils.to_categorical(ids, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


classifier = Sequential()

classifier.add(Dense(39, init='uniform', activation='relu', input_dim=39))
classifier.add(Dropout(0.2))
classifier.add(Dense(20, init='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(30, init='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(20, init='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(30, init='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(3, init='uniform', activation='sigmoid'))
classifier.add(Activation('softmax'))  # This special "softmax" a

classifier.compile(
    optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = classifier.fit(
    X_train, y_train, validation_split=0.33, epochs=100, batch_size=10)

y_pred = classifier.predict(X_test)

y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]


y_test = [np.argmax(y, axis=None, out=None) for y in y_test]


print("-------------------------")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print("-------------------------")
print("Accuracy Score: ")
print(accuracy_score(y_test, y_pred))


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
