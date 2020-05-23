# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:51:39 2020

@author: donsp

This program has been writen by Hanzhe Ye, all right reserved.

"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
from __future__ import print_function
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jieba
import jieba.posseg as pseg
from scipy import stats as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn import metrics
from IPython import display
import nltk
from nltk.corpus import stopwords
import matplotlib.font_manager as fm

tf.enable_eager_execution();
tf.logging.set_verbosity(tf.logging.ERROR);

pd.options.display.max_colwidth = 30;

training_data = pd.read_excel("training.xlsx");
validation_data = pd.read_excel("validation.xlsx");
display.display(training_data);


def preprocess_features(training_raw):  
  training_data = list(training_raw);
  
  words = []
  for sentence in training_data:
      words.append(sentence.split(" "));

  return words;

def preprocess_targets(labels):
  num_l=pd.Categorical(labels);
  codes = num_l.codes;
  categories = num_l.categories;
  return codes, categories;

training_set = preprocess_features(training_data["data"]);
training_tags, training_names = preprocess_targets(training_data["labels"]);
validation_set = preprocess_features(validation_data["data"]);
validation_tags, validation_names = preprocess_targets(validation_data["labels"]);
print(training_set,training_tags);

print(training_names);

def input_fn(features, targets, batch_size=5, num_epoch=None, shuffle=True):

  def package_generator():
    for feature, tag in zip(features, targets):
      yield feature, tag

  dataset = tf.data.Dataset.from_generator(package_generator, output_types=(tf.string,tf.int64), output_shapes=((None,),()))
  
  if shuffle == True:
    dataset.shuffle(90)

  dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
  dataset = dataset.repeat(num_epoch)

  next_feature, next_label = dataset.make_one_shot_iterator().get_next()
  return {"terms":next_feature}, next_label

sample_feature, sample_label = input_fn(training_set, training_tags)
print(sample_feature, sample_label)

categorical_column = tf.feature_column.categorical_column_with_hash_bucket("terms", 100000)

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, clip_norm=5.0)

terms_embedding_column = tf.feature_column.embedding_column(categorical_column, dimension=16)
feature_columns = [terms_embedding_column]

dnn_regressor = tf.estimator.DNNClassifier(
    hidden_units = [20,20],
    feature_columns = feature_columns,
    n_classes = 3,
    optimizer = my_optimizer
)

_ = dnn_regressor.train(
        input_fn = lambda: input_fn(training_set, training_tags),
        steps = 5000
    )

estimation = dnn_regressor.evaluate(
    input_fn = lambda: input_fn(validation_set, validation_tags),
    steps = 5000
)

print("training assessment:")
for name in estimation:
  print(name, estimation[name])
print("------")

predictions = dnn_regressor.predict(input_fn = lambda: input_fn(
                                                      validation_set,
                                                      validation_tags,
                                                      num_epoch=1,
                                                      shuffle=False))

predictions = np.array([item["probabilities"] for item in predictions])
predict_tag = []

for values in predictions:
  for i in range(len(values)):
    if values[i] == max(values):
      predict_tag.append(validation_names[i])

prediction_dataframe = pd.DataFrame()
prediction_dataframe["data"] = validation_data["data"]
prediction_dataframe["prediction"] = predict_tag
display.display(prediction_dataframe.head())
prediction_dataframe.to_excel("predictions.xlsx", sheet_name="results")
