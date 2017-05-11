import os,sys
import numpy as np
import tensorflow as tf
from InputPipeline import DataReader
from singleTask import CNNSingle

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, "training.txt", "testing.txt")

network = CNNSingle(data, 50, -1) #batch size, landmark
# network.debugNetwork()
sess = network.trainNetwork(25, 1.0) # epochs, keep_prob
network.testNetwork(sess)
network.outputImages(sess)
