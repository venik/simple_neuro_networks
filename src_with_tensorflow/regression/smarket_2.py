#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pandas as pd

# good example
# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/03_logistic_regression_mnist_sol.py

# constants
LEARNING_RATE = 0.01
EPOCHS = 100
batch_size = 100
display_step = 1

# prepare data
data_all = pd.read_csv('../../data_set/islr_datasets/Smarket.csv', header = 0)
data_train = data_all[data_all['Year'] < 2005]
data_test = data_all[data_all['Year'] == 2005]

train_X = np.asarray(data_train.loc[:, ('Lag1')])
train_Y = np.asarray(data_train.loc[:, ('Direction')].apply(lambda x: 'Up' in x).astype(int))

X = tf.placeholder(tf.float32, [batch_size, 1],name='X_placeholder')
Y = tf.placeholder(tf.int32, [batch_size, 1], name='Y_placeholder')

W = tf.Variable(tf.random_normal(shape=[1, 1], stddev=0.01), 'weights')
b = tf.Variable(tf.zeros([1, 1]), 'bias')

# model
pred = tf.matmul(X, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

# gradient descent
# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

batches = int(train_X.shape[0] / batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        total_loss = 0

        for kk in range(batches):
            X_batch = train_X[kk*batch_size : (kk+1)*batch_size].reshape(batch_size, 1)
            Y_batch = train_Y[kk*batch_size : (kk+1)*batch_size].reshape(batch_size, 1)

            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, total_loss / batches))

    print('Optimization Finished!')

    #################
    # Testing example
    preds = tf.nn.softmax(pred)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds = 0

    test_X = np.asarray(data_test.loc[:, ('Lag1')])
    test_Y = np.asarray(data_test.loc[:, ('Direction')].apply(lambda x: 'Up' in x).astype(int))
    test_batches = int(test_X.shape[0] / batch_size)

    for kk in range(test_batches):
        X_test = test_X[kk * batch_size : (kk + 1) * batch_size].reshape(batch_size, 1)
        Y_test = test_Y[kk * batch_size : (kk + 1) * batch_size].reshape(batch_size, 1)

        accuracy_batch = sess.run([accuracy], feed_dict={X: X_test, Y: Y_test})
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / test_X.shape[0]))