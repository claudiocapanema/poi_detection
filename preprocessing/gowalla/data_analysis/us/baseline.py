import pandas as pd
import numpy as np
import tensorflow as tf
from configuration import CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US
import json
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier


def input_fn_train(x_train, y_train):
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    x_train = tf.constant(x_train)
    y_train = tf.constant(y_train)
    train = (x_train, y_train)
    return train


def input_fn_eval(x_test, y_test):
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    x_test = tf.constant(x_test)
    x_test = tf.constant(y_test)
    test = (x_test, y_test)
    return test

if __name__ == "__main__":

    df = pd.read_csv(CHECKINS_LOCAL_DATETIME_COLUMNS_REDUCED_US)
    print(df)

    categories_list = df['category'].tolist()
    categories_flatten_list = categories_list

    x = [i for i in range(len(categories_flatten_list))]
    y = categories_flatten_list
    train = int(len(x) * 0.8)
    x_train = x[:train]
    y_train = y[:train]
    x_test = x[train:]
    y_test = y[train:]

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(x_train, y_train)
    predictions = dummy_clf.predict(x_test)

    print(predictions)

    report = classification_report(y_test, predictions)

    print(report)

    df = pd.DataFrame({1: [report]})
    df.to_csv("baseline.txt", header=False, index=False)





