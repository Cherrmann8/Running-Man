from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_ACT = 4

def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_2 = tf.tanh(tf.add(tf.matmul(_X, _weights['h2']), _biases['b2']))
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

def loss(logits, labels):
  cost = tf.reduce_sum(tf.pow(logits-labels, 2))/64
  return cost

def training(loss, learning_rate):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  return optimizer


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
