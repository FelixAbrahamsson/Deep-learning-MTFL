import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

class InputPipeline():
    def __init__(self, path, file):
        self.image_path = path
        self.num_data = 0

        
    def read_labeled_info(self, file):
        folder = self.image_path + "/"
        f = open(folder + file, "r")
        lines = f.readlines()
        fileNames = []
        landmarks = []
        attributes = []
        for line in lines:
            line = line.strip("\n ").split(" ")
            fileName = line[0].replace("\\", "/")
            if fileName == "":
                break
            fileNames.append(folder + fileName)
            coords = []
            for i in range(1,6):
                coords.append([float(line[i]),float(line[i+5])])
            attributes.append([int(line[i]) for i in range(11,15)])
            landmarks.append(coords)
        return fileNames, landmarks, attributes

    def get_input_que(self, file):
        image_list, landmark_list, attribute_list = self.read_labeled_info(file)
        images_tensor = ops.convert_to_tensor(image_list, dtype=dtypes.string)
        landmark_tensor = ops.convert_to_tensor(landmark_list, dtype=dtypes.float32)
        attribute_tensor = ops.convert_to_tensor(attribute_list, dtype=dtypes.int32)
        #produces an error when the number of epochs is exhausted
        input_que = tf.train.slice_input_producer([image_list, landmark_list, attribute_list], num_epochs=None)
        self.num_data = len(image_list)
        return input_que

    def read_from_disk(self, file):
        input_que = self.get_input_que(file)
        file_contents = tf.read_file(input_que[0])
        images = tf.image.decode_jpeg(file_contents, channels=3)/255
        #This line is needed since tf.train.batch needs to know the size of the tensor which tf.image.decode_jpeg strangley dosen't produce
        #causes an error for images with other sizes
        images.set_shape((150,150, 3))
        landmarks = input_que[1]
        attributes = input_que[2]
        return images, landmarks, attributes




class DataReader():
    def __init__(self, path, info_train, info_test):
        train_pipe = InputPipeline(path, info_train)
        train_img, train_land, train_att = train_pipe.read_from_disk(info_train)
        self.num_train_data = train_pipe.num_data
        self.train = [train_img, train_land, train_att]

        test_pipe = InputPipeline(path, info_test)
        test_img, test_land, test_att = test_pipe.read_from_disk(info_test)
        self.test = [test_img, test_land, test_att]
        self.num_test_data = test_pipe.num_data

    def read_batch(self, batch_size, train = True): #set to false if tou want to read the test set
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        if(train):
            print("reading from train set")
            image_batch, landmark_batch, attribute_batch= tf.train.shuffle_batch(
            self.train, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=3)
        else:
            print("reading from test set")
            image_batch, landmark_batch, attribute_batch= tf.train.shuffle_batch(
            self.test, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=3, allow_smaller_final_batch=True)
        return image_batch, landmark_batch, attribute_batch
    
