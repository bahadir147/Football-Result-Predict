import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def evaluate(model, test_x, test_y, train_x, train_y):

    y_pred = model.predict(test_x)

    # Train and Test Accuracy
    print("Train Accuracy :: ", accuracy_score(
        train_y,  model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, y_pred))
    print(" Confusion matrix ")
    print(confusion_matrix(test_y, y_pred))
