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
        self.f_size = 5 # receptive field size
        self.conv1filters = 16 # nr of output channels from conv layer 1
        self.conv2filters = self.conv1filters*2 # nr of output channels from conv layer 2
        self.fc1size = 1024

        if self.landmark == -1:
            self.output_size = 10
        else:
            self.output_size = 2
        self.create_comp_graph()

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

    def feed_forward(self, x, y):
        h_conv1 = tf.nn.relu(self.conv2d(x, self.W_conv1)+self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 38 * 38 * self.conv2filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1)+self.b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2

        if self.landmark == -1:
            # If y_ is [5,2] l2diffs becomes a 5x1 vector with the vector distances
            y_vectors = tf.reshape(y_conv, [-1,5,2])
            l2diffs = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                y, y_vectors)),reduction_indices=2))
            loss = tf.reduce_mean(tf.reduce_sum(l2diffs,
                reduction_indices=1))
        else:
            # l2diff will just be scalar values in this case
            y_vectors = y_conv
            l2diffs = tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                [y[:,self.landmark]], y_vectors)),reduction_indices=2)))
            loss = tf.reduce_mean(tf.squeeze(l2diffs))
        return l2diffs, y_vectors, loss

    def calc_accuracy(self, y, l2diffs):
        interocular_distance = tf.sqrt(tf.reduce_sum(tf.square(
          tf.subtract(y[:,0], y[:,1])),reduction_indices=1))
        # mean error is the l2 differences normalized by the interocular distance
        mean_error = l2diffs / tf.transpose([interocular_distance])
        # correct prediction if mean error < 10 %
        correct_prediction = tf.less(mean_error, 0.1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
        tf.float32),reduction_indices=0)
        return accuracy

    def initiate_net(self):
        # We have 3 input channels, rgb
        self.keep_prob = tf.placeholder(tf.float32)
        self.W_conv1 = self.weight_variable([self.f_size, self.f_size, 3, self.conv1filters])
        self.b_conv1 = self.bias_variable([self.conv1filters])

        self.W_conv2 = self.weight_variable([self.f_size, self.f_size, self.conv1filters, self.conv2filters])
        self.b_conv2 = self.bias_variable([self.conv2filters])

        # 38x38 = 150/2/2 x 150/2/2
        self.W_fc1 = self.weight_variable([38 * 38 * self.conv2filters, self.fc1size])
        self.b_fc1 = self.bias_variable([self.fc1size])

        self.W_fc2 = self.weight_variable([self.fc1size, self.output_size])
        self.b_fc2 = self.bias_variable([self.output_size])

    def create_comp_graph(self):     
        self.initiate_net()    

        self.train_x, self.train_y, self.train_attr = self.data.read_batch(self.batchSize, 0)
        self.train_l2diffs, self.train_y_vectors, self.train_loss =  self.feed_forward(self.train_x, self.train_y)
        self.train_acc = self.calc_accuracy(self.train_y, self.train_l2diffs)

        self.val_x, self.val_y, self.val_attr = self.data.read_batch(self.batchSize, 1)
        self.val_l2diffs, self.val_y_vectors, self.val_loss =  self.feed_forward(self.val_x, self.val_y)
        self.val_acc = self.calc_accuracy(self.val_y, self.val_l2diffs)

        self.test_x, self.test_y, self.test_attr = self.data.read_batch(self.batchSize, 2)
        self.test_l2diffs, self.test_y_vectors, self.test_loss =  self.feed_forward(self.test_x, self.test_y)
        self.test_acc = self.calc_accuracy(self.test_y, self.test_l2diffs)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.train_loss)

    def train_network(self, nrEpochs, keep_prob):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.size[0]//self.batchSize
        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1,nrEpochs):
            avg_acc = np.zeros(5)
            for i in range(steps):
                _, loss, acc = sess.run([self.train_step, self.train_loss, self.train_acc],
                    feed_dict={self.keep_prob:keep_prob})
                if (i == 0 and epoch == 1):
                    smooth_loss = loss
                else:
                    smooth_loss = 0.95 * smooth_loss + 0.05 * loss
                avg_acc = np.add(avg_acc, acc)
            val_acc = self.compute_accuracy_set(sess, 1)
            avg_acc = np.divide(avg_acc, steps)
            print("Epoch: " + str(epoch))
            print("Avg acc on training set: " + str(np.round(avg_acc,6)))
            print("Avg acc on validation set: " + str(val_acc))
            print("Smooth loss " + str(smooth_loss))
            # if epoch >= 10:
            #     self.testNetwork(sess)
        print("Training finished.")
        return sess

    def test_network(self, sess):
        mean_acc = self.compute_accuracy_set(sess, 2)
        print("Accuracy on test set: " + str(mean_acc))
     

    def compute_accuracy_set(self, sess, cur_set): # set i = [training, validation testing]
        mean_acc = np.zeros(5)
        steps = self.data.size[cur_set]//self.batchSize + 1
        for i in range(steps):
            if(cur_set == 1): # evaluate accuracy on validation set
                acc = sess.run([self.val_acc], feed_dict={self.keep_prob:1.0})[0]
            elif(cur_set == 2):
                acc = sess.run([self.test_acc], feed_dict={self.keep_prob:1.0})[0]
            mean_acc = np.add(mean_acc, acc)
        mean_acc = np.round(np.divide(mean_acc, steps), 6)
        return mean_acc
    
    def output_images(self, sess):
        radius = 2
        x, feature_vectors = sess.run([self.train_x, self.train_y_vectors],
            feed_dict={self.keep_prob:1.0})
        for i in range(5):
            img_mat = x[i]
            img_mat = np.multiply(img_mat, 255.0) # Scale back up
            img_data = img_mat.reshape(150*150,3).astype(int)
            img_data = tuple(map(tuple, img_data))
            im = Image.new("RGB", (150,150))
            im.putdata(img_data)
            draw = ImageDraw.Draw(im)
            if self.landmark == -1:
                y_ = feature_vectors[i]
            else:
                y_ = [feature_vectors[i]]
            for coords in y_:
                FL_x = coords[0]
                FL_y = coords[1]
                draw.ellipse((FL_x-radius, FL_y-radius, FL_x+radius, FL_y+radius), 
                    fill = 'green', outline ='blue')
            im.show()

    def debug_network(self):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        output = sess.run(self.loss, feed_dict={self.keep_prob:1.0, self.training:True})
        print(output)
