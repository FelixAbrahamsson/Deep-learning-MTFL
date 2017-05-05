import tensorflow as tf
import numpy as np
import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from InputPipeline import *

#init_op = tf.global_variables_initializer()
#init_op = init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#coord = tf.train.Coordinator()

    
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
    data = DataReader(dataFolder, "training.txt", "testing.txt")
    images, landmarks, attributes = data.read_batch(10)
    threads = tf.train.start_queue_runners(coord=coord)
    print(images)
    image = sess.run(images)
    print(image)
    
    print("hej")

