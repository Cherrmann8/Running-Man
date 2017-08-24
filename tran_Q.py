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
import matplotlib.pyplot as plt

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

EPOCHS = 1000
GAMMA = 0.9 #since it may take several moves to goal, making gamma high
TURN_BOUNDRY = 2000

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
    game = input.Input(3, 2)

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

    # Add to the Graph the Ops for loss calculation.
    loss = arch.loss(pred, action_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = arch.training(loss, FLAGS.learning_rate)

    # Build the summary operation based on the TF collection of Summaries.
    r = []
    l = []
    x = []
    t = []
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
    
    #ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    #if ckpt and ckpt.model_checkpoint_path:
    #  print('Loading ' + ckpt.model_checkpoint_path)
    #  # Restores from checkpoint
    #saver.restore(sess, "data/model_1.ckpt")
    model_num = 0 #int(ckpt.model_checkpoint_path.split("_")[1].split(".")[0]) + 1
    print(model_num)
    epsilon = 1
    wins = 0
    game_length = np.zeros(100)
    avg_score = np.zeros(100)
    xs = 0
    steps = 0
    avg_loss = np.zeros(100)
    avg_loss.fill(100)
    avg_score.fill(-100)
    print('Start Training...')
    # Start the training loop.
    for step in xrange(EPOCHS):
      #start_time = time.time()
      game.restart()
      status = 1
      turns = 0
      print("Game #: " + str(step), end="")
      while status == 1:
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        state = game.grid()
        qval = sess.run(pred, feed_dict = {state_placeholder : state.reshape(1, 75)})

        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,3)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        game.move(action)
        new_state = game.grid()
        #Observe reward
        reward = game.reward()

        #Get max_Q(S',a)
        newQ = sess.run(pred, feed_dict = {state_placeholder : new_state.reshape(1, 75)})
        maxQ = np.max(newQ)
        y = np.zeros((1,3))
        y[:] = qval[:]

        if reward < -1: #non-terminal state
            update = (reward + (GAMMA * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output

        feeddict = fill_feed_dict(state.reshape(1,75), y, state_placeholder, action_placeholder)
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feeddict)
        avg_loss[turns%100] = loss_value * 100
        turns += 1
        if game.total_out(-1) + game.pits != 3: 
          game.display()
          print('pits')
          print(turns)
        if game.total_out(1) + game.goals != 2:  
          game.display()
          print('goals')
          print(turns)
        if reward < -1 or turns >= TURN_BOUNDRY:
            status = 0

      print(" - Turns: %d" % (turns), end="")
      print(" - EC: %.2f" % (game.total_score))
      if len(game_length) >= 100:
        game_length[steps%100] = turns
        avg_score[steps%100] = game.total_score
      else:
        game_length.append(turns)
        avg_score.append(game.total_score)
      if turns >= 1000:
        wins += 1
      if epsilon > 0.001:
        epsilon -= (1/EPOCHS)

      #duration = time.time() - start_time

      if step == 0 or (step+1) % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f, avg game len = %.2f, avg score = %.2f, wins = %d' % (step, np.mean(avg_loss), np.mean(game_length), np.mean(avg_score), wins))
        # Update the events file.
        #summary_str = sess.run(summary_op, feed_dict=feeddict)
        #summary_writer.add_summary(summary_str, step)

      if step == 0 or (step+1) % 10 == 0:
        r.append(np.mean(avg_score))
        l.append(np.mean(avg_loss))
        t.append(np.mean(game_length))
        xs += 1
        x.append(xs*10)

      if (step+1) % (1000) == 0:
        save_path = saver.save(sess, "data/model_" + str(model_num) + ".ckpt")
        print("Model saved in file: %s" % save_path)
        model_num += 1
        turns = 0
        game.restart()
        status = 1
        while status == 1:
          game.display()
          #We are in state S
          #Let's run our Q function on S to get Q values for all possible actions
          state = game.grid()
          qval = sess.run(pred, feed_dict = {state_placeholder : state.reshape(1, 75)})

          if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,3)
            print('r')
          else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
            print('c')
          #Take action, observe new state S'
          game.move(action)
          new_state = game.grid()
          #Observe reward
          reward = game.reward()
          turns += 1
          if reward < -1 or turns >= 100: 
              status = 0
        game.display
        print(turns, game.total_score)

      steps += 1
      if wins > 0:
        r.append(np.mean(avg_score))
        l.append(np.mean(avg_loss))
        t.append(np.mean(game_length))
        xs += 1
        x.append(xs*10)
        break

  print(r, l, t, x)
  plt.plot(x, l, 'b')
  plt.plot(x, r, 'r')
  plt.plot(x, t, 'g')
  plt.axis([0, 1000, -100, 2000])
  plt.show()

def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
