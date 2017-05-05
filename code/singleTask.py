import numpy as np
import tensorflow as tf
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNSingle():

  def __init__(self, data, landmark):
    # landmark values:
    # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner
    self.data = data
    self.landmark = landmark
    self.createCompGraph(landmark)


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

  def createCompGraph(self, landmark):
    # landmark values: 
    # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner

    conv1filters = 16
    conv2filters = conv1filters*2

    if landmark == -1:
      output_size = 10
    else:
      output_size = 2

    self.x = tf.placeholder(tf.float32, shape=[None]+list(
      self.data.trainX[0].shape))
    # y_ are the coordinates of facial landmarks
    self.y_ = tf.placeholder(tf.float32, shape=[None, 5, 2]) 

    # Shape of W_conv1: [filter_height, filter_width, nr_channels, nr_filters]
    self.W_conv1 = self.weight_variable([5, 5, 3, conv1filters])
    self.b_conv1 = self.bias_variable([conv1filters]) 

    self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.W_conv1)+self.b_conv1)
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)

    self.W_conv2 = self.weight_variable([5, 5, conv1filters, conv2filters])
    self.b_conv2 = self.bias_variable([conv2filters])

    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1,
     self.W_conv2) + self.b_conv2)
    self.h_pool2 = self.max_pool_2x2(self.h_conv2)
    
    self.W_fc1 = self.weight_variable([38 * 38 * conv2filters, 1024])
    self.b_fc1 = self.bias_variable([1024])

    self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 38 * 38 * conv2filters])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1)+self.b_fc1)

    self.keep_prob = tf.placeholder(tf.float32)
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    self.W_fc2 = self.weight_variable([1024, output_size])
    self.b_fc2 = self.bias_variable([output_size])

    self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
    if landmark == -1:
      # If y_ is [5,2] l2diffs becomes a 5x1 vector with the vector distances
      self.y_vectors = tf.reshape(self.y_conv, [-1,5,2])
      self.l2diffs = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
        self.y_, self.y_vectors)),reduction_indices=2))
      self.loss = tf.reduce_mean(tf.reduce_sum(self.l2diffs,
        reduction_indices=1))
    else:
      # l2diff will just be scalar values in this case
      self.l2diffs = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
        [self.y_[:,landmark]], self.y_conv)),reduction_indices=2)))
      self.loss = tf.reduce_mean(tf.squeeze(self.l2diffs))

    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    # interocular distance is distance between eyes
    self.interocular_distance = tf.sqrt(tf.reduce_sum(tf.square(
      tf.subtract(self.y_[:,0], self.y_[:,1])),reduction_indices=1))
    # mean error is the l2 differences normalized by the interocular distance
    self.mean_error = self.l2diffs / tf.transpose([self.interocular_distance])
    # correct prediction if mean error < 10 %
    self.correct_prediction = tf.less(self.mean_error, 0.1)
    # accuracy either has shape (1) or shape (5)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 
      tf.float32),reduction_indices=0)


  def trainNetwork(self, nrSteps, batchSize):

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    for i in range(1,nrSteps+1):
      batch = self.data.nextBatch(batchSize)
      self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1],
       self.keep_prob: 0.5})
      if i%(math.ceil((nrSteps/10))) == 0:
        train_accuracy = self.evaluateAccuracy(batch[0], batch[1])
        print("Step %d, training accuracy: %s"%(i, train_accuracy))
        loss = self.evaluateLoss(batch[0],batch[1])
        print("Loss: %g"%loss)
    train_accuracy = self.evaluateAccuracy(batch[0], batch[1])
    print("Final training accuracy: %s"%train_accuracy)
    print("Training finished.")

  def evaluateAccuracy(self, x, y, rounding=6):
    return np.round(self.accuracy.eval(feed_dict={
      self.x:x, self.y_:y, self.keep_prob: 1.0}), rounding)

  def evaluateLoss(self, x, y):
    return self.loss.eval(feed_dict={
      self.x:x, self.y_:y, self.keep_prob: 1.0})
    # return sess.run(self.loss,feed_dict={
    #   self.x:x, self.y_:y, self.keep_prob: 1.0})

  def testNetwork(self):

    print("Evaluating network on test set...")
    mean_acc = [0,0,0,0,0]
    i = 0
    for i in range(200):
      if (i+1)*50 > self.data.testSize:
        break
      x = self.data.testX[i*50:(i+1)*50]
      y1 = self.data.testY1[i*50:(i+1)*50]
      acc = self.evaluateAccuracy(x, y1, rounding=10)
      for j in range(len(acc)):
        mean_acc[j] += acc[j]

    for j in range(len(mean_acc)):
      mean_acc[j] = mean_acc[j]/(i+1)
    mean_acc = np.round(mean_acc,6)
    
    if self.landmark == -1:
      print("Test accuracy: %s"%mean_acc)
    else:
      print("Test accuracy: %s"%mean_acc[0])

  def debugNetwork(self):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    batch = self.data.nextBatch(50)
    # batch = self.data.getTestdata()
    output = sess.run(self.accuracy, 
      feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
    print(output)