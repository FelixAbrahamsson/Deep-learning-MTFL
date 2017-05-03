import os,sys
import numpy as np
from PIL import Image
import pickle

class Dataset():

  def __init__(self, imageDir, trainTxt, testTxt, overwrite = False):
    self.imgSize = 150
    self.resizeImages = True
    self.trainXDataFile = 'trainX'
    self.trainYDataFile = 'trainY'
    self.testXDataFile = 'testX'
    self.testYDataFile = 'testY'
    self.dataDir = imageDir + '/processed/'
    self.imageDir = imageDir
    self.batchIndex = 0

    if not overwrite:
      print("Attempting to read training data from disk...")
      self.trainX, successX = self.readData(self.dataDir + self.trainXDataFile)
      self.trainY, successY = self.readData(self.dataDir + self.trainYDataFile)

    if overwrite or not successX or not successY:
      print("Training data not found on disk or overwrite selected, processing files...")
      self.trainX, self.trainY = self.processData(self.imageDir, trainTxt)
      print("Writing training data to disk...")
      self.storeData(self.trainXDataFile, self.trainX)
      self.storeData(self.trainYDataFile, self.trainY)

    if not overwrite:
      print("Attempting to read test data from disk...")
      self.testX, successX = self.readData(self.dataDir + self.testXDataFile)
      self.testY, successY = self.readData(self.dataDir + self.testYDataFile)

    if overwrite or not successX or not successY:
      print("Test data not found on disk or overwrite selected, processing files...")
      self.testX, self.testY = self.processData(self.imageDir, testTxt)
      print("Writing test data to disk...")
      self.storeData(self.testXDataFile, self.testX)
      self.storeData(self.testYDataFile, self.testX)

    print("Data loaded.")

    self.trainSize = len(self.trainX)
    self.testSize = len(self.trainY)


  def readData(self, path):
    try:
      fp = open(path, 'rb')
      data = pickle.load(fp)
      fp.close()
      return [data, True]
    except FileNotFoundError:
      return [None, False]

  def storeData(self, file_name,  data):
    if not os.path.exists(self.dataDir):
      os.makedirs(self.dataDir)
    
    with open(self.dataDir + file_name, 'wb') as fp:
      pickle.dump(data, fp)

  def processData(self, folder, txtfile):

    X = []
    Y = []
    folder = folder + "/"    
    f = open(folder+txtfile,"r")
    lines = f.readlines()
    counter = 0
    for line in lines:
      line = line.strip("\n ").split(" ")

      imgName = line[0].replace("\\", "/")
      if imgName[0:3] == "net":
        continue
      if imgName == "":
        break

      img = Image.open(folder+imgName, mode='r')
      originalWidth = img.width
      if self.resizeImages:
        img = img.resize((self.imgSize, self.imgSize), Image.ANTIALIAS)
      pixels = list(img.getdata())
      width, height = img.size
      pixelsArray = np.asarray([pixels[i * width:(i + 1) * width] for i in range(height)])
      pixelsArray = np.divide(pixelsArray, 255.0)
      X.append(pixelsArray)
      img.close()

      coords = []
      for i in range(1,6):
        coords.append((float(line[i]),float(line[i+5])))
      coordsScaleFactor = float(self.imgSize) / float(originalWidth)
      coords = np.multiply(np.array(coords), coordsScaleFactor)
      attributes = np.array([int(line[i]) for i in range(11,15)])
      Y.append([coords,attributes])

      counter += 1
      if counter % 1000 == 0:
        print(counter,"files read")

    f.close()
    print(counter,"files read total.")

    return [X,Y]

  def nextBatch(self, batchSize):
    batch = [None,None] # batch[0] is X, batch[1] is Y

    if self.batchIndex + batchSize >= self.trainSize:
      loopIndex = self.batchIndex + batchSize - self.trainSize
      batch[0] = self.trainX[self.batchIndex : self.trainSize] + self.trainX[0 : loopIndex]
      batch[1] = self.trainY[self.batchIndex : self.trainSize] + self.trainY[0 : loopIndex]
      self.batchIndex = loopIndex

    else:
      batch[0] = self.trainX[self.batchIndex : self.batchIndex + batchSize]
      batch[1] = self.trainY[self.batchIndex : self.batchIndex + batchSize]
      self.batchIndex = self.batchIndex + batchSize
    return batch

