#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pandas as pd
rng = np.random

# constants
LEARNING_RATE = 0.01
EPOCHS = 1
batch_size = 100
display_step = 1

# prepare data
data_all = pd.read_csv('../../data_set/islr_datasets/Smarket.csv', header = 0)
data_train = data_all[data_all['Year'] < 2005]
data_test = data_all[data_all['Year'] == 2005]

# data = data_train.loc[:, ('Lag1')]
# val = data_train.loc[:, ('Direction')].apply(lambda x: 'Up' in x).astype(int)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(0.0, 'weights')
b = tf.Variable(0.0, 'bias')

# model
pred = tf.add(tf.multiply(X, W), b)

# MSE cost function
cost = tf.reduce_sum(tf.pow(Y - pred, 2)) / data_train['Lag1'].shape[0]
# cost = tf.square(Y - pred, name="loss")

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_X = np.asarray(data_train.loc[:, ('Lag1')])
    train_Y = np.asarray(data_train.loc[:, ('Direction')].apply(lambda x: 'Up' in x).astype(int))
    for i in range(100):  # run 100 epochs
        for x, y in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

    ###############
    w_value, b_value = sess.run([W, b])
    print('w: ' + str(w_value) + ' b: ' + str(b_value))

    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #################
    # Testing example, as requested (Issue #2)
    test_Cost = tf.reduce_sum(tf.pow(Y - pred, 2)) / data_test['Lag1'].shape[0]

    test_X = np.asarray(data_test.loc[:, ('Lag1')])
    test_Y = np.asarray(data_test.loc[:, ('Direction')].apply(lambda x: 'Up' in x).astype(int))

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))