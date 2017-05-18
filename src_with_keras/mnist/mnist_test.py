#!/usr/bin/env python

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

train_data.astype('float32')
train_data /= 255

data_set = mnist('../../data_set/mnist', isTrainMode=False)
test_sample_num = data_set.get_num_of_frames_in_dataset()

test_data = np.zeros((test_sample_num, 784))
test_labels = np.zeros((test_sample_num, 1))
for i in range(0, test_sample_num):
    label, data = data_set.get_next_frame()
    test_data[i, :] = data
    test_labels[i] = label

test_data.astype('float32')
test_data /= 255

###############
b_size = 32
train_cat_labels = keras.utils.to_categorical(train_labels, 10)
model.fit(train_data, train_cat_labels, epochs=5, batch_size=b_size)

test_cat_labels = keras.utils.to_categorical(test_labels, 10)
score = model.evaluate(test_data, test_cat_labels, batch_size=b_size)

print('\n\nres: {:.2f} {:.2f}'.format(score[0], score[1]))

sample_id = 120
x = test_data[sample_id, :]
x = np.expand_dims(x, axis=0)
res = model.predict(x, batch_size=1, verbose=1)
label_prob = np.max(res)
label_id = np.where(res == label_prob)
print('label: ' + str(test_labels[sample_id]) + ' predicted: ' + str(label_id[1]) + ' with prob: {:.2f}'.format(label_prob))
