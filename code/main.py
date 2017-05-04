import os,sys
import numpy as np
import tensorflow as tf
from InputPipeline import DataReader
from singleTask import CNNSingle
from Dataset import Dataset

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = Dataset(dataFolder, "training.txt", "testing.txt")
# data = DataReader(dataFolder)
# print(data.trainX[0].shape)


network = CNNSingle(data)
# network.debugNetwork()
network.trainNetwork(100, 50)
network.testNetwork(100)

