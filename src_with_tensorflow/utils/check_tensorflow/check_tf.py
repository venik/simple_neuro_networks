#!/usr/bin/env python

import tensorflow as tf

a = tf.constant(6)
b = tf.constant(7)

sess = tf.Session()
print(sess.run(a * b))