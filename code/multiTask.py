import numpy as np
import tensorflow as tf
import math
import os
import time
import random
from PIL import Image
from PIL import ImageDraw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1:Info, 2:Warning, 3:Error

class CNNMulti():

    def __init__(self, data, batchSize):

        self.data = data
        self.batchSize = batchSize
        self.f_size = 5 # receptive field size
        self.conv1filters = 16 # nr of output channels from conv layer 1
        self.conv2filters = self.conv1filters*2 # nr of output channels from conv layer 2
        self.fc1size = 1024

        self.output_size_FL = 10
        self.output_size_attr1 = 2 # gender, glasses, smile
        self.output_size_attr2 = 5 # head pose

        # These control how much the different attribute loss functions
        # contribute to the total loss
        self.lambda_gender = 3.0
        self.lambda_smile = 3.0
        self.lambda_glasses = 3.0
        self.lambda_pose = 3.0

        self.create_comp_graph()

    def weight_variable(self, shape, name):
        initial = tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
        return initial
        # initial = tf.truncated_normal(shape, stddev=0.1)
        # return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
        return initial
        # initial = tf.constant(0.1, shape=shape)
        # return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    def shared_layer_output(self, x):
        h_conv1 = tf.nn.relu(self.conv2d(x, self.W_conv1)+self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 38 * 38 * self.conv2filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1)+self.b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        return h_fc1_drop

    def get_loss_FL(self, y_, y):
        # y_ is the input from the first FC layer

        y_conv = tf.matmul(y_, self.W_fc2_FL) + self.b_fc2_FL
        # If y_conv is [5,2] l2diffs becomes a 5x1 vector with the vector distances
        y_vectors = tf.reshape(y_conv, [-1,5,2])
        l2diffs = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
            y, y_vectors)),reduction_indices=2))
        loss = tf.reduce_mean(tf.reduce_sum(l2diffs,
            reduction_indices=1))

        return l2diffs, y_vectors, loss

    def get_loss_attr(self, y_, y, W_fc2, b_fc2):
        # y_ is the input from the first FC layer
        y_conv = tf.matmul(y_, W_fc2) + b_fc2
        loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
        return y_conv, loss

    def calc_accuracy_FL(self, y, l2diffs):
        interocular_distance = tf.sqrt(tf.reduce_sum(tf.square(
          tf.subtract(y[:,0], y[:,1])),reduction_indices=1))
        # mean error is the l2 differences normalized by the interocular distance
        mean_error = l2diffs / tf.transpose([interocular_distance])
        # correct prediction if mean error < 10 %
        correct_prediction = tf.less(mean_error, 0.1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
            tf.float32), reduction_indices=0)
        return accuracy

    def calc_accuracy_attr(self, y_conv, y):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.cast(y, tf.int64))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def initiate_net(self):
        # We have 3 input channels, rgb
        self.keep_prob = tf.placeholder(tf.float32)
        self.W_conv1 = self.weight_variable([self.f_size, self.f_size, 3, self.conv1filters], "W_c1")
        self.b_conv1 = self.bias_variable([self.conv1filters], "b_c1")

        self.W_conv2 = self.weight_variable([self.f_size, self.f_size, self.conv1filters, self.conv2filters], "W_c2")
        self.b_conv2 = self.bias_variable([self.conv2filters], "b_c2")

        # 38x38 = 150/2/2 x 150/2/2
        self.W_fc1 = self.weight_variable([38 * 38 * self.conv2filters, self.fc1size], "W_fc1")
        self.b_fc1 = self.bias_variable([self.fc1size], "b_fc1")

        self.W_fc2_FL = self.weight_variable([self.fc1size, self.output_size_FL], "W_fc2_FL")
        self.b_fc2_FL = self.bias_variable([self.output_size_FL], "b_fc2_FL")

        self.W_fc2_gender = self.weight_variable([self.fc1size, self.output_size_attr1], "W_fc2_ge")
        self.b_fc2_gender = self.bias_variable([self.output_size_attr1], "b_fc2_ge")

        self.W_fc2_smile = self.weight_variable([self.fc1size, self.output_size_attr1], "W_fc2_sm")
        self.b_fc2_smile = self.bias_variable([self.output_size_attr1], "b_fc2_sm")

        self.W_fc2_glasses = self.weight_variable([self.fc1size, self.output_size_attr1], "W_fc2_gl")
        self.b_fc2_glasses = self.bias_variable([self.output_size_attr1], "b_fc2_gl")

        self.W_fc2_pose = self.weight_variable([self.fc1size, self.output_size_attr2], "W_fc2_po")
        self.b_fc2_pose = self.bias_variable([self.output_size_attr2], "b_fc2_po")


    def get_attributes_sparse(self, attr):

        gender = attr[:,0]
        smile = attr[:,1]
        glasses = attr[:,2]
        pose = attr[:,3]
        return gender, smile, glasses, pose

    def create_comp_graph(self):     
        self.initiate_net()    

        # Training comp graph
        self.train_x, self.train_y, self.train_attr = self.data.read_batch(self.batchSize, 0)
        self.train_gender, self.train_smile, self.train_glasses, self.train_pose = self.get_attributes_sparse(self.train_attr)

        train_fc_1 = self.shared_layer_output(self.train_x)
        self.train_l2diffs, self.train_y_vectors, self.train_loss_FL = self.get_loss_FL(train_fc_1, self.train_y)
        self.train_acc_FL = self.calc_accuracy_FL(self.train_y, self.train_l2diffs)
        # Get losses and acc for attributes
        self.train_y_conv_gender, self.train_loss_gender = self.get_loss_attr(
            train_fc_1, self.train_gender, self.W_fc2_gender, self.b_fc2_gender)
        self.train_y_conv_smile, self.train_loss_smile = self.get_loss_attr(
            train_fc_1, self.train_smile, self.W_fc2_smile, self.b_fc2_smile)
        self.train_y_conv_glasses, self.train_loss_glasses = self.get_loss_attr(
            train_fc_1, self.train_glasses, self.W_fc2_glasses, self.b_fc2_glasses)
        self.train_y_conv_pose, self.train_loss_pose = self.get_loss_attr(
            train_fc_1, self.train_pose, self.W_fc2_pose, self.b_fc2_pose)
        self.train_acc_gender = self.calc_accuracy_attr(self.train_y_conv_gender, self.train_gender)
        self.train_acc_smile = self.calc_accuracy_attr(self.train_y_conv_smile, self.train_smile)
        self.train_acc_glasses = self.calc_accuracy_attr(self.train_y_conv_glasses, self.train_glasses)
        self.train_acc_pose = self.calc_accuracy_attr(self.train_y_conv_pose, self.train_pose)


        # Validation comp graph
        self.val_x, self.val_y, self.val_attr = self.data.read_batch(self.batchSize, 1)
        self.val_gender, self.val_smile, self.val_glasses, self.val_pose = self.get_attributes_sparse(self.val_attr)

        val_fc_1 = self.shared_layer_output(self.val_x)
        self.val_l2diffs, self.val_y_vectors, self.val_loss_FL = self.get_loss_FL(val_fc_1, self.val_y)
        self.val_acc_FL = self.calc_accuracy_FL(self.val_y, self.val_l2diffs)
        # Get losses and acc for attributes
        self.val_y_conv_gender, self.val_loss_gender = self.get_loss_attr(
            val_fc_1, self.val_gender, self.W_fc2_gender, self.b_fc2_gender)
        self.val_y_conv_smile, self.val_loss_smile = self.get_loss_attr(
            val_fc_1, self.val_smile, self.W_fc2_smile, self.b_fc2_smile)
        self.val_y_conv_glasses, self.val_loss_glasses = self.get_loss_attr(
            val_fc_1, self.val_glasses, self.W_fc2_glasses, self.b_fc2_glasses)
        self.val_y_conv_pose, self.val_loss_pose = self.get_loss_attr(
            val_fc_1, self.val_pose, self.W_fc2_pose, self.b_fc2_pose)
        self.val_acc_gender = self.calc_accuracy_attr(self.val_y_conv_gender, self.val_gender)
        self.val_acc_smile = self.calc_accuracy_attr(self.val_y_conv_smile, self.val_smile)
        self.val_acc_glasses = self.calc_accuracy_attr(self.val_y_conv_glasses, self.val_glasses)
        self.val_acc_pose = self.calc_accuracy_attr(self.val_y_conv_pose, self.val_pose)


        # Test comp graph
        self.test_x, self.test_y, self.test_attr = self.data.read_batch(self.batchSize, 2)
        self.test_gender, self.test_smile, self.test_glasses, self.test_pose = self.get_attributes_sparse(self.test_attr)

        test_fc_1 = self.shared_layer_output(self.test_x)
        self.test_l2diffs, self.test_y_vectors, self.test_loss_FL = self.get_loss_FL(test_fc_1, self.test_y)
        self.test_acc_FL = self.calc_accuracy_FL(self.test_y, self.test_l2diffs)
        # Get losses and acc for attributes
        self.test_y_conv_gender, self.test_loss_gender = self.get_loss_attr(
            test_fc_1, self.test_gender, self.W_fc2_gender, self.b_fc2_gender)
        self.test_y_conv_smile, self.test_loss_smile = self.get_loss_attr(
            test_fc_1, self.test_smile, self.W_fc2_smile, self.b_fc2_smile)
        self.test_y_conv_glasses, self.test_loss_glasses = self.get_loss_attr(
            test_fc_1, self.test_glasses, self.W_fc2_glasses, self.b_fc2_glasses)
        self.test_y_conv_pose, self.test_loss_pose = self.get_loss_attr(
            test_fc_1, self.test_pose, self.W_fc2_pose, self.b_fc2_pose)
        self.test_acc_gender = self.calc_accuracy_attr(self.test_y_conv_gender, self.test_gender)
        self.test_acc_smile = self.calc_accuracy_attr(self.test_y_conv_smile, self.test_smile)
        self.test_acc_glasses = self.calc_accuracy_attr(self.test_y_conv_glasses, self.test_glasses)
        self.test_acc_pose = self.calc_accuracy_attr(self.test_y_conv_pose, self.test_pose)

        # Calculate total loss function to be minimized by the training step
        self.stop_FL = tf.placeholder(tf.bool)
        self.stop_gender = tf.placeholder(tf.bool)
        self.stop_smile = tf.placeholder(tf.bool)
        self.stop_glasses = tf.placeholder(tf.bool)
        self.stop_pose = tf.placeholder(tf.bool)

        self.loss_FL_contr = tf.where(self.stop_FL, 0.0, self.train_loss_FL)
        self.loss_gender_contr = tf.where(self.stop_gender, 0.0, self.train_loss_gender)
        self.loss_smile_contr = tf.where(self.stop_smile, 0.0, self.train_loss_smile)
        self.loss_glasses_contr = tf.where(self.stop_glasses, 0.0, self.train_loss_glasses)
        self.loss_pose_contr = tf.where(self.stop_pose, 0.0, self.train_loss_pose)

        self.total_train_loss = self.loss_FL_contr + self.lambda_gender * self.loss_gender_contr + \
            self.lambda_smile * self.loss_smile_contr + self.lambda_glasses * self.loss_glasses_contr + \
            self.lambda_pose * self.loss_pose_contr

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.total_train_loss)

    def train_network(self, nrEpochs, keep_prob):
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        steps = self.data.size[0]//self.batchSize
        stop_FL = False
        stop_gender = False
        stop_smile = False
        stop_glasses = False
        stop_pose = False

        print("Number of steps per epoch: " + str(steps))
        for epoch in range(1, nrEpochs + 1):
            train_mean_acc_FL = np.zeros(5)
            train_mean_acc_attr = np.zeros(4)
            for i in range(steps):
                acc_attr = np.zeros(4)
                _, acc_FL, acc_attr[0], acc_attr[1], acc_attr[2], acc_attr[3] = sess.run(
                    [self.train_step, self.train_acc_FL, self.train_acc_gender, 
                    self.train_acc_smile, self.train_acc_glasses, self.train_acc_pose],
                    feed_dict={self.keep_prob:keep_prob, self.stop_FL:stop_FL, 
                    self.stop_gender:stop_gender, self.stop_smile:stop_smile, 
                    self.stop_glasses:stop_glasses, self.stop_pose:stop_pose})
                train_mean_acc_FL = np.add(train_mean_acc_FL, acc_FL)
                train_mean_acc_attr = np.add(train_mean_acc_attr, acc_attr)
            train_mean_acc_FL = np.round(np.divide(train_mean_acc_FL, steps), 6)
            train_mean_acc_attr = np.round(np.divide(train_mean_acc_attr, steps), 6)

            val_acc_FL, val_acc_attr, val_losses = self.compute_accuracy_set(sess, 1)
            print("Epoch: " + str(epoch))
            print("FL mean acc on train set: " + str(train_mean_acc_FL))
            print("Attributes mean acc on train set: " + str(train_mean_acc_attr))
            print("FL acc on validation set: " + str(val_acc_FL))
            print("Attributes acc on validation set: " + str(val_acc_attr))
            print("Losses on validation set (FL, ge, sm, gl, po): " + str(val_losses))
        print("Training finished.")
        return sess

    def test_network(self, sess):
        mean_acc_FL, mean_acc_attr, losses = self.compute_accuracy_set(sess, 2)
        print("FL acc on test set: " + str(mean_acc_FL))
        print("Attributes acc on test set: " + str(mean_acc_attr))
        print("Losses on test set (FL, ge, sm, gl, po): " + str(losses))

    def compute_accuracy_set(self, sess, cur_set): # set i = [training, validation testing]
        mean_losses = np.zeros(5)
        mean_acc_FL = np.zeros(5)
        mean_acc_attr = np.zeros(4)
        steps = self.data.size[cur_set]//self.batchSize
        for i in range(steps):
            acc_attr = np.zeros(4)
            losses = np.zeros(5)
            if(cur_set == 1): # evaluate accuracy on validation set
                losses[0], losses[1], losses[2], losses[3], losses[4], \
                acc_FL, acc_attr[0], acc_attr[1], acc_attr[2], acc_attr[3] = sess.run([self.val_loss_FL,
                    self.val_loss_gender, self.val_loss_smile, self.val_loss_glasses, self.val_loss_pose,
                    self.val_acc_FL, self.val_acc_gender, self.val_acc_smile, self.val_acc_glasses, self.val_acc_pose], 
                    feed_dict={self.keep_prob:1.0})
            elif(cur_set == 2):
                losses[0], losses[1], losses[2], losses[3], losses[4], \
                acc_FL, acc_attr[0], acc_attr[1], acc_attr[2], acc_attr[3] = sess.run([self.test_loss_FL,
                    self.test_loss_gender, self.test_loss_smile, self.test_loss_glasses, self.test_loss_pose,
                    self.test_acc_FL, self.test_acc_gender, self.test_acc_smile, self.test_acc_glasses, self.test_acc_pose], 
                    feed_dict={self.keep_prob:1.0})
            mean_acc_FL = np.add(mean_acc_FL, acc_FL)
            mean_acc_attr = np.add(mean_acc_attr, acc_attr)
            mean_losses = np.add(mean_losses, losses)
        mean_acc_FL = np.round(np.divide(mean_acc_FL, steps), 6)
        mean_acc_attr = np.round(np.divide(mean_acc_attr, steps), 6)
        mean_losses = np.round(np.divide(mean_losses, steps), 6)
        return mean_acc_FL, mean_acc_attr, mean_losses
    
    def output_images(self, sess):
        radius = 2.0
        x, feature_vectors = sess.run([self.test_x, self.test_y_vectors],
            feed_dict={self.keep_prob:1.0})
        for i in range(5):
            img_mat = x[i]
            img_mat = np.multiply(img_mat, 255.0) # Scale back up
            img_data = img_mat.reshape(150*150,3).astype(int)
            img_data = tuple(map(tuple, img_data))
            im = Image.new("RGB", (150,150))
            im.putdata(img_data)
            draw = ImageDraw.Draw(im)

            for coords in feature_vectors[i]:
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

        output = sess.run(self.train_pose, 
            feed_dict={self.keep_prob:1.0})
        # print(output)

        fl, ge, sm, gl, po = sess.run([self.train_loss_FL, self.train_loss_gender, self.train_loss_smile, self.train_loss_glasses, self.train_loss_pose], 
            feed_dict={self.keep_prob:1.0})
        print(fl)
        print(ge)
        print(sm)
        print(gl)
        print(po)
