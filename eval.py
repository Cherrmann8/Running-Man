# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import arch
import input
import random
import numpy as np


EPOCHS = 10

def placeholder_inputs():
  states_placeholder = tf.placeholder(tf.float32, shape=(None, 75))
  actions_placeholder = tf.placeholder(tf.float32, shape=(None, 3))
  return states_placeholder, actions_placeholder


def fill_feed_dict(state, actions, states_pl, actions_pl):

  feed_dict = {
      states_pl: state,
      actions_pl: actions,
  }
  return feed_dict


def run_training():
  """Train MNIST for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    state_placeholder, action_placeholder = placeholder_inputs()
    game = input.Input(6, 1)

    def multilayer_perceptron(_X, _weights, _biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
        return tf.matmul(layer_2, _weights['out']) + _biases['out']
    # Store layers weight & bias
    weights = { 
        'h1': tf.Variable(tf.random_normal([75, 75])),
        'h2': tf.Variable(tf.random_normal([75, 40])),
        'out': tf.Variable(tf.random_normal([40, 3]))
    }
    biases = { 
        'b1': tf.Variable(tf.random_normal([75])),
        'b2': tf.Variable(tf.random_normal([40])),
        'out': tf.Variable(tf.random_normal([3]))
    }

    # Construct model
    pred = multilayer_perceptron(state_placeholder, weights, biases)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    saver.restore(sess, "data/model_2.ckpt")
    epsilon = 1

    print('Start Training...')
    # Start the training loop.
    for step in xrange(EPOCHS):
      turns = 0
      game.restart()
      status = 1
      while status == 1:
        state = game.grid()
        qval = sess.run(pred, feed_dict = {state_placeholder : state.reshape(1, 75)})

        action = (np.argmax(qval))
        game.move(action)
        new_state = game.grid()
        reward = game.reward()
        turns += 1
        if turns % 1000 == 0:
          print(turns)
        if reward < -1 or turns >= 10000: 
          status = 0
      print(turns, game.total_score)


def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
