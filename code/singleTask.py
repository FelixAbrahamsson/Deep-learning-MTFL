import numpy as np
import tensorflow as tf
import math
import os
import time
from PIL import Image
from PIL import ImageDraw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNSingle():

    def __init__(self, data, batchSize, landmark):
        # landmark values:
        # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner
        self.data = data
        self.landmark = landmark
        self.batchSize = batchSize
        self.createCompGraph()
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


    def createCompGraph(self):
        # landmark values:
        # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner

        f_size = 5 # receptive field size
        conv1filters = 16 # nr of output channels from conv layer 1
        conv2filters = conv1filters*2 # nr of output channels from conv layer 2
        fc1size = 1024

        if self.landmark == -1:
            output_size = 10
        else:
            output_size = 2

        with tf.variable_scope("Input"):
            self.training = tf.placeholder(tf.bool)
            self.train_x, self.train_y_, self.train_attr = self.data.read_batch(
                self.batchSize, True)
            self.test_x, self.test_y_, self.test_attr = self.data.read_batch(
                self.batchSize, False)

        with tf.variable_scope("Net"):

            # Choose between testing and training
            self.x = tf.where(self.training, self.train_x, self.test_x)
            self.y_ = tf.where(self.training, self.train_y_, self.test_y_)
            self.attr = tf.where(self.training, self.train_attr, self.test_attr)

            # We have 3 input channels, rgb
            self.W_conv1 = self.weight_variable([f_size, f_size, 3, conv1filters])
            self.b_conv1 = self.bias_variable([conv1filters])

            self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.W_conv1)+self.b_conv1)
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)

            self.W_conv2 = self.weight_variable([f_size, f_size, conv1filters, conv2filters])
            self.b_conv2 = self.bias_variable([conv2filters])

            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1,
             self.W_conv2) + self.b_conv2)
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)
            
            # 38x38 = 150/2/2 x 150/2/2
            self.W_fc1 = self.weight_variable([38 * 38 * conv2filters, fc1size])
            self.b_fc1 = self.bias_variable([fc1size])

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 38 * 38 * conv2filters])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1)+self.b_fc1)

            self.keep_prob = tf.placeholder(tf.float32)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            self.W_fc2 = self.weight_variable([fc1size, output_size])
            self.b_fc2 = self.bias_variable([output_size])

            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        with tf.variable_scope("Loss"):
            if self.landmark == -1:
                # If y_ is [5,2] l2diffs becomes a 5x1 vector with the vector distances
                self.y_vectors = tf.reshape(self.y_conv, [-1,5,2])
                self.l2diffs = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                    self.y_, self.y_vectors)),reduction_indices=2))
                self.loss = tf.reduce_mean(tf.reduce_sum(self.l2diffs,
                    reduction_indices=1))
            else:
                # l2diff will just be scalar values in this case
                self.y_vectors = self.y_conv
                self.l2diffs = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                    [self.y_[:,self.landmark]], self.y_vectors)),reduction_indices=2)))
                self.loss = tf.reduce_mean(tf.squeeze(self.l2diffs))

        with tf.variable_scope("Accuracy"):
            # interocular distance is distance between eyes
            self.interocular_distance = tf.sqrt(tf.reduce_sum(tf.square(
              tf.subtract(self.y_[:,0], self.y_[:,1])),reduction_indices=1))
            # mean error is the l2 differences normalized by the interocular distance
            self.mean_error = self.l2diffs / tf.transpose([self.interocular_distance])
            # correct prediction if mean error < 10 %
            self.correct_prediction = tf.less(self.mean_error, 0.1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 
                tf.float32),reduction_indices=0)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def trainNetwork(self, nrEpochs, keep_prob):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.num_train_data//self.batchSize
        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1,nrEpochs+1):
            avg_acc = np.zeros(5)
            for i in range(steps):
                _, loss, acc = sess.run([self.train_step, self.loss, self.accuracy],
                    feed_dict={self.keep_prob:keep_prob, self.training:True})
                if (i == 0 and epoch == 1):
                    smooth_loss = loss
                else:
                    smooth_loss = 0.95 * smooth_loss + 0.05 * loss
                avg_acc = np.add(avg_acc, acc)
            avg_acc = np.divide(avg_acc, steps)
            print("Epoch: " + str(epoch))
            print("Avg acc on training set: " + str(np.round(avg_acc,6)))
            print("Smooth loss " + str(smooth_loss))
            # if epoch >= 20:
            #     self.testNetwork(sess)
        print("Training finished.")
        return sess

    def testNetwork(self, sess):
        mean_acc = np.zeros(5)
        steps = self.data.num_test_data//self.batchSize
        for i in range(steps):
            acc = sess.run([self.accuracy], 
                feed_dict={self.keep_prob:1.0, self.training:False})[0]
            mean_acc = np.add(mean_acc, acc)
        mean_acc = np.round(np.divide(mean_acc, steps), 6)
        if self.landmark != -1:
            mean_acc = mean_acc[0]

        print("Accuracy on test set: " + str(mean_acc))

    def outputImages(self, sess):
        radius = 2
        # x = sess.run([self.x])
        x, feature_vectors = sess.run([self.x, self.y_vectors],
            feed_dict={self.keep_prob:1.0, self.training:False})
        for i in range(5):
            imgMat = x[i]
            imgMat = np.multiply(imgMat, 255.0) # Scale back up
            imgData = imgMat.reshape(150*150,3).astype(int)
            imgData = tuple(map(tuple, imgData))
            im = Image.new("RGB", (150,150))
            im.putdata(imgData)
            draw = ImageDraw.Draw(im)
            for coords in feature_vectors[i]:
                FL_x = coords[0]
                FL_y = coords[1]
                draw.ellipse((FL_x-radius, FL_y-radius, FL_x+radius, FL_y+radius), 
                    fill = 'green', outline ='blue')
            im.show()

    def debugNetwork(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # batch = self.data.nextBatch(50)
        batch = self.data.getTestdata()
        output = sess.run(self.x, 
          feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        print(output.shape)
