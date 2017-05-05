import os,sys
import numpy as np
import tensorflow as tf
from InputPipeline import DataReader
from singleTask import CNNSingle
from Dataset import Dataset

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, "training.txt", "testing.txt")
#data = Dataset(dataFolder, "training.txt", "testing.txt")
# data = DataReader(dataFolder)
# print(data.trainX[0].shape)


network = CNNSingle(data, 100) #batch size
# network.debugNetwork()
sess = network.trainNetwork(20) # epochs
network.testNetwork(sess)

