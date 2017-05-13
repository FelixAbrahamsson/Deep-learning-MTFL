import numpy as np
import tensorflow as tf
import math
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNSingleAtt():

    def __init__(self, data, batchSize, attribute):
        # attribute value:
        # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner
        self.data = data
        self.batchSize = batchSize
        self.createCompGraph(attribute)
        self.shape = [150, 150, 3]
    
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

    def createCompGraph(self, attribute):
        # landmark values:
        # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner

        conv1filters = 16
        conv2filters = conv1filters*2

        if attribute < 3:
            output_size = 2
        else:
            output_size = 5

        with tf.variable_scope("Input"):
            self.training = tf.placeholder(tf.bool)
            self.train_x, self.train_land, self.train_y_ = self.data.read_batch(self.batchSize, True)
            self.test_x, self.test_land, self.test_y_ = self.data.read_batch(self.batchSize, False)

        with tf.variable_scope("Net"):

            # Choose between testing and training
            self.x = tf.where(self.training, self.train_x, self.test_x)
            self.y_gt = tf.where(self.training, self.train_y_, self.test_y_)
            self.y_ = tf.one_hot(self.y_gt[:,attribute], output_size)

            self.land = tf.where(self.training, self.train_land, self.test_land)

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

            self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        with tf.variable_scope("Loss"):
            self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.y_ * self.y_conv, axis=1)+0.0000000001))

        with tf.variable_scope("Accuracy"):
            self.accuracy = tf.reduce_mean(tf.reduce_sum(self.y_conv * self.y_, axis=1))

        
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def trainNetwork(self, nrEpochs, keep_prob):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.num_train_data//self.batchSize
        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1,nrEpochs+1):
            avg_acc = 0.0
            for i in range(steps):
                _, loss, acc = sess.run([self.train_step, self.loss, self.accuracy],
                    feed_dict={self.keep_prob:keep_prob, self.training:True})
                if (i == 0 and epoch == 1):
                    smooth_loss = loss
                else:
                    smooth_loss = 0.95 * smooth_loss + 0.05 * loss
                avg_acc += acc
            avg_acc = avg_acc / steps
            print("Epoch: " + str(epoch))
            print("Accuracy on training set: " + str(avg_acc))
            print("Smooth loss " + str(smooth_loss))
        print("Training finished.")
        return sess

    def testNetwork(self, sess):
        self.x, landmarks, self.y_ = self.data.read_batch(self.batchSize, False)
        mean_acc = 0
        steps = self.data.num_test_data//self.batchSize
        for i in range(steps):
            acc = sess.run([self.accuracy], feed_dict={self.keep_prob:1.0, self.training:False})[0]
            mean_acc += acc
        mean_acc = mean_acc/steps

        print("Accuracy on test set: " + str(mean_acc))

    def debugNetwork(self):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        output = sess.run(self.loss, feed_dict={self.keep_prob:1.0, self.training:True})
        print(output)
