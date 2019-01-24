# Kütüphaneler.
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
import keras
import numpy as np
import sklearn
from keras import optimizers
import talos as ta
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
from keras.layers import *
from keras.models import *
from time import time


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

nb_classes = 10

X_train = X_train.reshape(6766, 39)
X_test = X_test.reshape(3333, 39)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)




#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit(y_train)
#y_train = enc.transform(y_train)




print(X_train)

print(y_train)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(3, input_shape=(39,)))
    # An "activation" is just a non-linear function applied to the output
    model.add(Activation('relu'))
    # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dropout(0.2))
    model.add(Dense(512, init=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, init=init))
    model.add(Activation('softmax'))  # This special "softmax" a
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


start = time()
model = KerasClassifier(build_fn=create_model)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs,
                  batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose=1)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




########################################################DÜZENELEME
# Use scikit-learn to grid search 
activation =  ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] # softmax, softplus, softsign 
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_constraint=[1, 2, 3, 4, 5]
neurons = [1, 5, 10, 15, 20, 25, 30]
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
##############################################################
# grid search epochs, batch size
epochs = [1, 10] # add 50, 100, 150 etc
batch_size = [1000, 5000] # add 5, 10, 20, 40, 60, 80, 100 etc
param_grid = dict(epochs=epochs, batch_size=batch_size)