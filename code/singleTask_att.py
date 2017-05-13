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

        output_size = 1

        with tf.variable_scope("Input"):
            self.training = tf.placeholder(tf.bool)
            self.train_x, self.train_land, self.train_y_ = self.data.read_batch(self.batchSize, True)
            self.test_x, self.test_land, self.test_y_ = self.data.read_batch(self.batchSize, False)

        # with tf.variable_scope("Test"):
        #     images, landmark, attributes = self.data.read_batch(self.num_test_data)
        #     self.x_test = images
        #     self.y_test = landmark
        # # self.x = tf.placeholder(tf.float32, shape=[None,150,150,3])
        # # y_ are the coordinates of facial landmarks
        # self.y_ = tf.placeholder(tf.float32, shape=[None, 5, 2]) 

        with tf.variable_scope("Net"):

            # Choose between testing and training
            self.x = tf.where(self.training, self.train_x, self.test_x)
            self.y = tf.where(self.training, self.train_y_, self.test_y_)
            self.y_ = tf.to_float(tf.add(self.y, 1), name='ToFloat')
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

            self.y_conv = tf.nn.softmax(tf.reduce_sum(tf.transpose( tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2), axis=0))
            #self.y_conv = tf.Print(self.y_conv, [self.y_conv])

        with tf.variable_scope("Loss"):
            #self.l2diffs = tf.square(tf.subtract([self.y_[:,attribute]], ))
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_[:, attribute] * tf.log( self.y_conv )))

            self.ex1 = self.y_[:,attribute]
            self.ex2 =  self.y_conv[:]

        with tf.variable_scope("Accuracy"):
            #self.accuracy = tf.constant(0)
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv) , tf.argmax(self.y_[:,attribute]) )
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            #for i in range(self.batchSize):
                

                #self.condition1 = tf.logical_and( tf.equal(self.y_[i, attribute],10) ,tf.greater(self.y_conv[i], 0))    
                #self.condition2 = tf.logical_and(tf.equal(self.y_[i, attribute],-10), tf.less_equal(self.y_conv[i], 0))
                #self.cond = tf.logical_or(self.condition1, self.condition2)
                #self.incr = tf.cast(self.cond, tf.int32)
                #self.accuracy = tf.add(self.incr, self.accuracy)
            
            #self.accuracy = tf.divide(self.accuracy, self.batchSize)
        
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def trainNetwork(self, nrEpochs, keep_prob):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.num_train_data//self.batchSize
        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1,nrEpochs+1):
            for i in range(steps):
                _ = sess.run([self.train_step],feed_dict={self.keep_prob:keep_prob, self.training:True})
            loss, acc, ex1, ex2 = sess.run([self.loss, self.accuracy, self.ex1, self.ex2], feed_dict={self.keep_prob:1.0, self.training:True})
            print("Examples: \n"+str(ex1)+"\n"+str(ex2))
            print("Epoch: " + str(epoch))
            print("Accuracy on training set: " + str(acc))
            print("Loss " + str(loss))
        print("Training finished.")
        return sess

    def testNetwork(self, sess):
        self.x, landmarks, self.y_ = self.data.read_batch(self.batchSize, False)
        mean_acc = 0 #tf.constant(0)
        steps = self.data.num_test_data//self.batchSize
        for i in range(steps):
            acc = sess.run([self.accuracy], feed_dict={self.keep_prob:1.0, self.training:False})[0]
            mean_acc += acc #tf.add(self.mean_acc, acc)
        mean_acc = mean_acc/steps#tf.divide(self.mean_acc, steps)

        print("Accuracy on test set: " + str(mean_acc))

    def debugNetwork(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        batch = self.data.read_batch(50, train = False)
        #batch = self.data.getTestdata()
        output = sess.run(self.x, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        print(output.shape)
