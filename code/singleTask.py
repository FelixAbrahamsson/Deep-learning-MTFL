import numpy as np
import tensorflow as tf
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNSingle():

  def __init__(self, data):
    self.data = data
    self.createCompGraph()


  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  def createCompGraph(self):

    self.x = tf.placeholder(tf.float32, shape=[None]+list(
      self.data.trainX[0].shape))
    # y_ are the coordinates of facial landmarks
    self.y_ = tf.placeholder(tf.float32, shape=[None, 5, 2]) 

    self.W_conv1 = self.weight_variable([5, 5, 3, 32])
    self.b_conv1 = self.bias_variable([32]) 

    self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.W_conv1)+self.b_conv1)
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)

    self.W_conv2 = self.weight_variable([5, 5, 32, 64])
    self.b_conv2 = self.bias_variable([64])

    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1,
     self.W_conv2) + self.b_conv2)
    self.h_pool2 = self.max_pool_2x2(self.h_conv2)
    
    self.W_fc1 = self.weight_variable([38 * 38 * 64, 1024])
    self.b_fc1 = self.bias_variable([1024])

    self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 38 * 38 * 64])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1)+self.b_fc1)

    self.keep_prob = tf.placeholder(tf.float32)
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    self.W_fc2 = self.weight_variable([1024, 10])
    self.b_fc2 = self.bias_variable([10])

    self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
    self.y_vectors = tf.reshape(self.y_conv, [-1,5,2])

    # if y_ is [5,2] l2diffs becomes a 5x1 vector with the vector distances
    self.l2diffs = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
      self.y_, self.y_vectors)),reduction_indices=2))
    self.loss = tf.reduce_sum(self.l2diffs,reduction_indices=1)
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    # interocular distance is distance between eyes
    self.interocular_distance = tf.sqrt(tf.reduce_sum(tf.square(
      tf.subtract(self.y_[:,0,:], self.y_[:,1,:])),reduction_indices=1))
    # mean error is the l2 differences normalized by the interocular distance
    self.mean_error = self.l2diffs / tf.transpose([self.interocular_distance])
    # correct prediction if mean error < 10 %
    self.correct_prediction = tf.less(self.mean_error, 0.1)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

  def trainNetwork(self, nrEpochs, batchSize):

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    for i in range(nrEpochs):
      batch = self.data.nextBatch(batchSize)
      if i%(math.ceil((nrEpochs/10))) == 0:
        train_accuracy = self.accuracy.eval(feed_dict={
            self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
        print("Step %d, training accuracy %g"%(i, train_accuracy))
      self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1],
       self.keep_prob: 0.5})
    print("Done training!")


  def testNetwork(self):
    batch = self.data.getTestdata()
    train_accuracy = self.accuracy.eval(feed_dict={
            self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
    print(train_accuracy)
    # print("test accuracy %g"%self.accuracy.eval(feed_dict={
    #   self.x:self.data.testX, self.y_:self.data.testY1, self.keep_prob:1.0}))

  def debugNetwork(self):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # batch = self.data.nextBatch(50)
    batch = self.data.getTestdata()
    output = sess.run(self.x, 
      feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
    print(output.shape)