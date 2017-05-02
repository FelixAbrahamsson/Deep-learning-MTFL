"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os,sys

from tensorflow.examples.tutorials.mnist import input_data
import pickle
import tensorflow as tf
from Dataset import Dataset

FLAGS = None

def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
  data = Dataset(dataFolder, "training.txt", "testing.txt")

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  h_conv1 = tf.nn.relu(conv2d(data.trainX, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 14])
  b_fc2 = bias_variable([14])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.global_variables_initializer())
  for i in range(10):
    batchX, batchY = data.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batchX, y_: batchY, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: data.testX, y_: data.testY, keep_prob: 1.0}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)