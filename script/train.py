#!/usr/bin/env python
# coding: utf-8

"""
Train process with rendered videos (standarized)
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, SpatialDropout3D, MaxPooling3D, Conv3D, LeakyReLU, ReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2

tf.logging.set_verbosity(tf.logging.ERROR)

def stand(img):
    return (img - np.mean(img))/np.std(img)

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])

def get_train_data(s, std=False):
    if std:
        sd = "_std"
    else:
        sd = ""
    
    if s == "xyz":
        X = np.load("../data/X"+sd+".npy")
        y = np.load("../data/y.npy")
    else:
        X = np.load("../data/"+s+"/X_train"+sd+".npy")
        y = np.load("../data/"+s+"/y_train.npy")
    
    return X, y

def get_test_data(s, std=False):
    if std:
        sd = "_std"
    else:
        sd = ""
    
    X_test = np.load("../data/"+s+"/X_test"+sd+".npy")
    y_test = np.load("../data/"+s+"/y_test.npy")
    
    return X_test, y_test

def train():
    X_shape = [12, 128, 128, 1]

    model = Sequential()

    model.add(Conv3D(16, 5, padding='same', input_shape = X_shape, data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size = (2,2,2)))

    model.add(Conv3D(16, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size = (2,2,2)))

    model.add(Conv3D(16, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size = (1,2,2)))

    model.add(Conv3D(16, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size = (1,2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #opt = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1 ,patience=2)
    checkpoint = ModelCheckpoint("../output/model/new/model-{val_acc:.2f}.hdf5",
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [reduceLR, checkpoint]

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    X, y = get_train_data("xyz")
    X_test, y_test = get_test_data("y")

    X_index = np.reshape(list(range(len(X))), (len(X), 1))
    y_index = np.argmax(y, axis=1)

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_index, y_index)
    new_index = [X_resampled[i].tolist()[0] for i in range(len(X_resampled))]

    X = X[new_index]
    y = get_one_hot(y_resampled, 10)

    history = model.fit(X, y, epochs=20, batch_size=5, validation_data=(X_test, y_test), callbacks=callbacks_list)

    with open('../output/history/hist.pickle', 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_train():
    with open('../output/history/hist.pickle', 'rb') as handle:
        hist = pickle.load(handle)

    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    train()
    plot_train()

if __name__ == "__main__":
    main()