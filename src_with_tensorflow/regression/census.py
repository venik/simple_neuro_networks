#!/usr/bin/env python

import pandas as pd
import tensorflow as tf

import tempfile

# example from
# https://www.tensorflow.org/tutorials/wide

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital_status', 'occupation',
                       'relationship', 'race', 'gender', 'native_country']

CONTINUOUS_COLUMNS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def test_input_fn():
    return input_fn(df_test)

df_train = pd.read_csv('../../data_set/census/adult.data', names = COLUMNS, skipinitialspace=True)
df_test = pd.read_csv('../../data_set/census/adult.test', names = COLUMNS, skipinitialspace=True, skiprows=1)

df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

gender = tf.contrib.layers.sparse_column_with_keys(column_name='gender', keys=['female', 'male'])

education = tf.contrib.layers.sparse_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
    "relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
    "workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# Continuous base columns.
age = tf.contrib.layers.real_valued_column("age")

education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

# Transformations.
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Wide columns and deep columns.
wide_columns = [gender, native_country, education, occupation, workclass,
                relationship, age_buckets,
                tf.contrib.layers.crossed_column([education, occupation],
                                                hash_bucket_size=int(1e4)),
                tf.contrib.layers.crossed_column(
                    [age_buckets, education, occupation],
                    hash_bucket_size=int(1e6)),
                tf.contrib.layers.crossed_column([native_country, occupation],
                                                hash_bucket_size=int(1e4))]

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(
    model_dir=model_dir,
    feature_columns=wide_columns)
    # optimizer=tf.train.FtrlOptimizer(
    #     learning_rate=0.1,
    #     l1_regularization_strength=0.5,
    #     l2_regularization_strength=0.5))

m.fit(input_fn=lambda: input_fn(df_train), steps=200)
results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


