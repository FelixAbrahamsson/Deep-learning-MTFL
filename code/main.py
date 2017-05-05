import os,sys
import numpy as np
import tensorflow as tf
from singleTask import CNNSingle
from Dataset import Dataset

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = Dataset(dataFolder, "training.txt", "testing.txt", 2000)

network = CNNSingle(data, -1)
network.trainNetwork(100, 50)
network.testNetwork()

# network.debugNetwork()

