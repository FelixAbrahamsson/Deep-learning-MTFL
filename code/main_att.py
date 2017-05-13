import os,sys
import numpy as np
import tensorflow as tf
from InputPipeline import DataReader
from singleTask_att import CNNSingleAtt

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, "training.txt", "testing.txt")


network = CNNSingleAtt(data, 80, 0) #batch size, landmark
#network.debugNetwork()
sess = network.trainNetwork(10, 1.0) # epochs, keep_prob
network.testNetwork(sess)

