#!/usr/bin/env python
# coding: utf-8

"""
Test process with rendered videos (standarized)
Author: Ruochenliu
Date: June 2019
"""

import numpy as np
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
from os import listdir
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.logging.set_verbosity(tf.logging.ERROR)

def get_test_data(s, std=True):
    if std:
        sd = "_std"
    else:
        sd = ""
    
    X_test = np.load("../data/"+s+"/X_test"+sd+".npy")
    y_test = np.load("../data/"+s+"/y_test.npy")
    
    return X_test, y_test

def test():
    model = load_model('../output/model/model-best.hdf5')
    X_x, y_x = get_test_data("x")
    X_y, y_y = get_test_data("y")
    X_z, y_z = get_test_data("z")
    predict_x = model.predict(X_x)
    predict_y = model.predict(X_y)
    predict_z = model.predict(X_z)
    pre = (predict_x + predict_y + predict_z)/3
    real = y_x
    test_acc = np.mean(np.argmax(pre, 1) == np.argmax(real, 1))
    print('Test accuracy: {}'.format(test_acc))

def main():
    test()

if __name__ == "__main__":
    main()