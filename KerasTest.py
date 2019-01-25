# Kütüphaneler.
import Accuracy
import DataSaveLoad
import GetData
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
import keras
import numpy as np
import sklearn
from keras import optimizers
from keras.layers import *
from keras.models import *
import os
import matplotlib.pyplot as plt
import keras.callbacks as cb
from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
from keras.utils import np_utils
from matplotlib import pyplot as plt
import time
import pandas as pd

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#!/usr/bin/python


TunnigData = False


Datas = GetData.GetData('Data\GlobalData.xlsx')

X_train = Datas[0]
y_train = Datas[1]
X_test = Datas[2]
y_test = Datas[3]

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

################Pre-processing###########
x_train = preprocessing.scale(X_train)
x_test = preprocessing.scale(X_test)


activation_func = 'relu'
loss_function = 'categorical_crossentropy'
#loss_function = 'mean_squared_error'

dropout_rate = 0.4
weight_regularizer = None
learning_rate = 0.005

# Initialize model.
model = Sequential()
# 1st Layer
# Dense' means fully-connected.
model.add(Dense(128, input_dim=39, W_regularizer=weight_regularizer))
model.add(Activation(activation_func))
model.add(Dropout(0.5))

# 2nd Layer
model.add(Dense(64, input_dim=128, W_regularizer=weight_regularizer))
model.add(Activation(activation_func))
model.add(Dropout(dropout_rate))

# 3rd Layer
model.add(Dense(32))
model.add(Activation(activation_func))
model.add(Dropout(dropout_rate))

# 4th Layer
model.add(Dense(16))
model.add(Activation(activation_func))
model.add(Dropout(dropout_rate))

# 5th Layer
model.add(Dense(8))
model.add(Activation(activation_func))
model.add(Dropout(dropout_rate))

# Adding Softmax Layer
# Last layer has the same dimension as the number of classes
model.add(Dense(4))

# For classification, the activation is softmax
model.add(Activation('softmax'))

# Define optimizer. we select Adam
opt = Adam(lr=learning_rate, beta_1=0.9,
           beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = SGD(lr=learning_rate, clipnorm=5.)

# Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])


batch = 128
start_time = time.time()
epochs = 500

# Use the first 55,000 (out of 60,000) samples to train, last 5,500 samples to validate.
history = model.fit(x_train, y_train, nb_epoch=epochs,
                    batch_size=batch, validation_split=0.2)
print("Training took {0} seconds.".format(time.time() - start_time))


trained_model = model
training_history = history


def PlotHistory(train_value, test_value, value_is_loss_or_acc):
    f, ax = plt.subplots()
    ax.plot([None] + train_value, 'o-')
    ax.plot([None] + test_value, 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train ' + value_is_loss_or_acc,
               'Validation ' + value_is_loss_or_acc], loc=0)
    ax.set_title('Training/Validation ' + value_is_loss_or_acc + ' per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value_is_loss_or_acc)


PlotHistory(training_history.history['loss'],
            training_history.history['val_loss'], 'Loss')
PlotHistory(training_history.history['acc'],
            training_history.history['val_acc'], 'Accuracy')


def drawWeightHistogram(x):
    # the histogram of the data
    fig = plt.subplots()
    n, bins, patches = plt.hist(x, 50)
    plt.xlim(-0.5, 0.5)
    plt.xlabel('Weight')
    plt.ylabel('Count')
    zero_counts = (x == 0.0).sum()
    plt.title("Weight Histogram. Num of '0's: %d" % zero_counts)


w1 = trained_model.layers[0].get_weights()[0].flatten()
drawWeightHistogram(w1)


def TestModel(model=None, data=None):
    if model is None:
        print("Must provide a trained model.")
        return
    if data is None:
        print("Must provide data.")
        return
    x_test, y_test = data
    scores = model.evaluate(x_test, y_test)
    return scores


test_score = TestModel(model=trained_model, data=[x_test, y_test])

print("Test loss {:.4f}, accuracy {:.2f}%".format(
    test_score[0], test_score[1] * 100))
