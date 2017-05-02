import os,sys
import numpy as np
import tensorflow as tf
from Dataset import Dataset

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = Dataset(dataFolder, "training.txt", "testing.txt")
print(data.trainX[0][0][0])
