## This file can be used to just run the network quickly, without
## command line arguments


import os,sys
import numpy as np
from InputPipeline import DataReader
from singleTask import CNNSingle
from singleTask_att import CNNSingleAtt
from multiTask import CNNMulti

train_txt = "training.txt"
# train_txt = "aug_training.txt"

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])
print_data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])

# network = CNNSingle(data, print_data, 50, -1) #batch size, landmark
# network = CNNSingleAtt(data, 50, 1) #batch size, attribute
network = CNNMulti(data, print_data, 50) #batch size

# network.debug_network()
sess = network.train_network(1, 1.0, True) # epochs, keep_prob, use_early_stopping
network.test_network(sess)
network.output_images(sess, "multi_es_")