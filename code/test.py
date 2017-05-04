import tensorflow as tf
import numpy as np
import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from InputPipeline import *

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
    data = DataReader(dataFolder, "training.txt", "testing.txt", num_epochs = None)
    images, landmarks, attributes = data.read_batch(10)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
        x =images.eval() #here is your image Tensor :)
        y = landmarks.eval()
        print(y.shape)
        print(x.shape)

    coord.request_stop()
    coord.join(threads)
    sess.close()
