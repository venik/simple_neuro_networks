#!/usr/bin/env python

import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf

# example from ISLR chapter 4, classification

FEATURES = ['Lag1', 'Lag2', 'Lag3']#, 'Lag4', 'Lag5']
LABEL = 'label'

def input_data(data):
    feature_cols = {k: tf.constant(data[k].values) for k in FEATURES}
    cols = dict(feature_cols.items())

    labels = tf.constant(data[LABEL].values)

    return cols, labels


data_all = pd.read_csv('../../data_set/islr_datasets/Smarket.csv', header = 0)

data_train = data_all[data_all['Year'] < 2005]
data_test = data_all[data_all['Year'] == 2005]

print(str(data_all.size))
print(str(data_train.size))
print(str(data_test.size))

# print(str(data_all.columns))
# print(str(data_test.iloc[0]))
# print(str(data_train.iloc[0]))

# print(str(data_test['Direction'].values))

# categorical
# up_down = tf.contrib.layers.sparse_column_with_keys(column_name = 'Direction', keys = ['Down', 'Up'])

# data_train[LABEL] = (data_train['Direction'].apply(lambda x: 'Up' in x)).astype(int)
# data_test[LABEL] = (data_train['Direction'].apply(lambda x: 'Up' in x)).astype(int)

data_train[LABEL] = data_train.loc[:, 'Direction'].apply(lambda x: 'Up' in x).astype(int)
data_test[LABEL] = data_test.loc[:, 'Direction'].apply(lambda x: 'Up' in x).astype(int)

# # print(data_train[LABEL])

# continous
lag1 = tf.contrib.layers.real_valued_column(column_name = 'Lag1')
lag2 = tf.contrib.layers.real_valued_column(column_name = 'Lag2')
lag3 = tf.contrib.layers.real_valued_column(column_name = 'Lag3')
lag4 = tf.contrib.layers.real_valued_column(column_name = 'Lag4')
lag5 = tf.contrib.layers.real_valued_column(column_name = 'Lag5')
volume = tf.contrib.layers.real_valued_column(column_name = 'Volume')
today = tf.contrib.layers.real_valued_column(column_name = 'Today')

# bucktized_lag1 = tf.contrib.layers.bucketized_column(lag1, boundaries=np.linspace(-10, 10, 10).tolist())
# bucktized_lag2 = tf.contrib.layers.bucketized_column(lag2, boundaries=np.linspace(-10, 10, 10).tolist())
# bucktized_lag3 = tf.contrib.layers.bucketized_column(lag3, boundaries=np.linspace(-10, 10, 10).tolist())


# feature_columns = [lag1, lag2, lag3]#, lag4, lag5]
feature_columns = [	lag1,
					lag2,
					lag3,
					tf.contrib.layers.crossed_column([lag1, lag2], hash_bucket_size=int(1e6))]

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(
    model_dir = model_dir,
    feature_columns = feature_columns
)

m.fit(input_fn=lambda: input_data(data_train), steps=200)

## check results
results = m.evaluate(input_fn=lambda: input_data(data_test), steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

print('===================')
for var in m.get_variable_names():
    print(var + ': ' + str(m.get_variable_value(var)))