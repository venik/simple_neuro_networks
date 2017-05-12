#!/bin/env python

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
# model.add(Activation('softmax'))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

import sys
sys.path.append('../../')
from lib.mnist.mnist import mnist
import numpy as np

data_set = mnist('../../data_set/mnist')
train_sample_num = data_set.get_num_of_frames_in_dataset()
train_data = np.zeros((train_sample_num, 784))
train_labels = np.zeros((train_sample_num, 1))
for i in range(0, train_sample_num):
    label, data = data_set.get_next_frame()
    train_data[i, :] = data
    train_labels[i] = label

train_data /= 255

data_set = mnist('../../data_set/mnist', isTrainMode=False)
test_sample_num = data_set.get_num_of_frames_in_dataset()

test_data = np.zeros((test_sample_num, 784))
test_labels = np.zeros((test_sample_num, 1))
for i in range(0, test_sample_num):
    label, data = data_set.get_next_frame()
    test_data[i, :] = data
    test_labels[i] = label

test_data /= 255

###############
b_size = 32
train_cat_labels = keras.utils.to_categorical(train_labels, 10)
model.fit(train_data, train_cat_labels, epochs=1, batch_size=b_size)

test_cat_labels = keras.utils.to_categorical(test_labels, 10)
score = model.evaluate(test_data, test_cat_labels, batch_size=b_size)

print('\n\nres: {:.2f} {:.2f}'.format(score[0], score[1]))
