import os,sys
import numpy as np
from InputPipeline import DataReader
from singleTask import CNNSingle
from singleTask_att import CNNSingleAtt
from multiTask import CNNMulti

args = sys.argv
trial_nr = int(args[1])
network_type = int(args[2])
use_aug = int(args[3])
attribute = int(args[4])
use_es = int(args[5])
attribute_str = {0:"gender",1:"smile",2:"glasses",3:"pose"}
train_txt = "training.txt"

if use_aug == 0:
  train_txt = "training.txt"
  print("====== No augmentation =======")
if use_aug == 1:
  print("====== With augmentation =======")
  train_txt = "aug_training.txt"

if use_es == 1:
  print("====== With early stopping =======")

dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])
print_data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])

if network_type == 0:
  net_str = "single"
  network = CNNSingle(data, print_data, 50, -1) #batch size, landmark
elif network_type == 1:
  net_str = "single_"+str(attribute_str[attribute])
  network = CNNSingleAtt(data, 50, attribute) #batch size, attribute
elif network_type == 2:
  net_str = "multi"
  network = CNNMulti(data, print_data, 50) #batch size
else:
  print("ERROR: INCORRECT NETWORK TYPE")

if use_aug == 1:
  net_str += "_aug"
if use_es == 1:
  net_str += "_es"

print("====== "+str(net_str)+" network, trial "+str(trial_nr)+" ========")

sess = network.train_network(30, 1.0, bool(use_es)) # epochs, keep_prob, use_early_stopping
network.test_network(sess)
if network_type != 1:
  network.output_images(sess, net_str+str(trial_nr)+"_")
sess.close()


### Just to run the network without command line inputs
# train_txt = "training.txt"
# train_txt = "aug_training.txt"

# dataFolder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL")
# data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])
# print_data = DataReader(dataFolder, [train_txt, "validation.txt", "testing.txt"])

# network = CNNSingle(data, print_data, 50, -1) #batch size, landmark
# network = CNNSingleAtt(data, 50, 1) #batch size, attribute
# network = CNNMulti(data, print_data, 50) #batch size

# network.debug_network()
# sess = network.train_network(1, 1.0, True) # epochs, keep_prob
# network.test_network(sess)
# network.output_images(sess, "multi_es1_")