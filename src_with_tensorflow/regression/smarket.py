#!/usr/bin/env python

import pandas as pd
import tempfile
import tensorflow as tf

# example from ISLR chapter 4, classification

FEATURES = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
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

# print(data_train[LABEL])

# continous
lag1 = tf.contrib.layers.real_valued_column(column_name = 'Lag1')
lag2 = tf.contrib.layers.real_valued_column(column_name = 'Lag2')
lag3 = tf.contrib.layers.real_valued_column(column_name = 'Lag3')
lag4 = tf.contrib.layers.real_valued_column(column_name = 'Lag4')
lag5 = tf.contrib.layers.real_valued_column(column_name = 'Lag5')
volume = tf.contrib.layers.real_valued_column(column_name = 'Volume')
today = tf.contrib.layers.real_valued_column(column_name = 'Today')

feature_columns = [lag1, lag2, lag3, lag4, lag5, volume]
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
