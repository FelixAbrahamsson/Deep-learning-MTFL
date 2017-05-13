import os,sys
import numpy as np
import tensorflow as tf
from InputPipeline import DataReader
from singleTask_att import CNNSingleAtt


train_txt = "training.txt"
# train_txt = "aug_training.txt"

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])


network = CNNSingleAtt(data, 50, 2) #batch size, landmark
# network.debugNetwork()
sess = network.trainNetwork(3, 1.0) # epochs, keep_prob
network.testNetwork(sess)

