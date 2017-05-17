import numpy as np
import tensorflow as tf
import math
import os
import time
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNSingleAtt():

    def __init__(self, data, batchSize, attribute):
        # landmark values:
        # -1:all 0:l_eye 1:r_eye 2:nose 3:l_mouth_corner 4:r_mouth_corner

        self.data = data
        self.batchSize = batchSize
        self.attribute = attribute
        self.f_size = 5 # receptive field size
        self.conv1filters = 16 # nr of output channels from conv layer 1
        self.conv2filters = self.conv1filters*2 # nr of output channels from conv layer 2
        self.fc1size = 1024

        if self.attribute < 3:
            self.output_size = 2
        else:
            self.output_size = 5
        self.create_comp_graph()

    def weight_variable(self, shape, name):
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()) 
        return w

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    def feed_forward(self, x, y):
        h_conv1 = tf.nn.relu(self.conv2d(x, self.W_conv1)+self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 38 * 38 * self.conv2filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1)+self.b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
        return y_conv, loss

    def calc_accuracy(self, y, y_conv):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def initiate_net(self):
        # We have 3 input channels, rgb
        self.keep_prob = tf.placeholder(tf.float32)
        self.W_conv1 = self.weight_variable([self.f_size, self.f_size, 3, self.conv1filters], "w1_conv")
        self.b_conv1 = self.bias_variable([self.conv1filters])

        self.W_conv2 = self.weight_variable([self.f_size, self.f_size, self.conv1filters, self.conv2filters], "w2_conv")
        self.b_conv2 = self.bias_variable([self.conv2filters])

        # 38x38 = 150/2/2 x 150/2/2
        self.W_fc1 = self.weight_variable([38 * 38 * self.conv2filters, self.fc1size], "w1_fc")
        self.b_fc1 = self.bias_variable([self.fc1size])

        self.W_fc2 = self.weight_variable([self.fc1size, self.output_size], "w2_fc")
        self.b_fc2 = self.bias_variable([self.output_size])

    def create_comp_graph(self):     
        self.initiate_net()    

        self.train_x, _ , self.train_attr = self.data.read_batch(self.batchSize, 0)
        self.train_attr = tf.one_hot(self.train_attr[:,self.attribute], self.output_size)
        self.train_attr_conv, self.train_loss =  self.feed_forward(self.train_x, self.train_attr)
        self.train_acc = self.calc_accuracy(self.train_attr, self.train_attr_conv)

        self.val_x, _ , self.val_attr = self.data.read_batch(self.batchSize, 1)
        self.val_attr = tf.one_hot(self.val_attr[:,self.attribute], self.output_size)
        self.val_attr_conv, self.val_loss =  self.feed_forward(self.val_x, self.val_attr)
        self.val_acc = self.calc_accuracy(self.val_attr, self.val_attr_conv)

        self.test_x, _, self.test_attr = self.data.read_batch(self.batchSize, 2)
        self.test_attr = tf.one_hot(self.test_attr[:,self.attribute], self.output_size)
        self.test_attr_conv, self.test_loss =  self.feed_forward(self.test_x, self.test_attr)
        self.test_acc = self.calc_accuracy(self.test_attr, self.test_attr_conv)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.train_loss)

    def train_network(self, nrEpochs, keep_prob, dummyVar):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.size[0]//self.batchSize
        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1, nrEpochs + 1):
            avg_acc = 0
            for i in range(steps):
                _, loss, acc, conv = sess.run([self.train_step, self.train_loss, self.train_acc, self.train_attr_conv],
                    feed_dict={self.keep_prob:keep_prob})
                if (i == 0 and epoch == 1):
                    smooth_loss = loss
                else:
                    smooth_loss = 0.95 * smooth_loss + 0.05 * loss
                avg_acc += acc
            val_acc = self.compute_accuracy_set(sess, 1)
            avg_acc = avg_acc/steps
            print("Epoch: " + str(epoch))
            # print("Avg acc on training set: " + str(np.round(avg_acc,6)))
            print("Avg acc on validation set: " + str(val_acc))
            # print("Smooth loss " + str(smooth_loss))
        print("Training finished.")
        return sess

    def test_network(self, sess):
        mean_acc = self.compute_accuracy_set(sess, 2)
        print("Accuracy on test set: " + str(mean_acc))
     

    def compute_accuracy_set(self, sess, cur_set): # set i = [training, validation testing]
        mean_acc = 0
        steps = self.data.size[cur_set]//self.batchSize + 1
        for i in range(steps):
            if(cur_set == 1): # evaluate accuracy on validation set
                acc = sess.run([self.val_acc], feed_dict={self.keep_prob:1.0})[0]
            elif(cur_set == 2):
                acc = sess.run([self.test_acc], feed_dict={self.keep_prob:1.0})[0]
            mean_acc += acc
        mean_acc = mean_acc/steps
        return mean_acc
    

    def debug_network(self):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        output = sess.run(self.loss, feed_dict={self.keep_prob:1.0, self.training:True})
        print(output)
