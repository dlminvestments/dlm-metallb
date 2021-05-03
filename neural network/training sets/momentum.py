# nn_momentum.py
# Python 3.x

import numpy as np
import random
import math

# ------------------------------------

def showVector(v, dec): ...
def showMatrixPartial(m, numRows, dec, indices): ...

# ------------------------------------

class NeuralNetwork: ...
  
# ------------------------------------

def main():
  print("\nBegin NN back-propagation with momentum demo \n")
  
  numInput = 4
  numHidden = 5
  numOutput = 3
  seed = 3
  print("Creating a %d-%d-%d neural network " %
   (numInput, numHidden, numOutput) )
  nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
  
  print("\nLoading Iris training and test data ")
  trainDataPath = "irisTrainData.txt"
  trainDataMatrix = np.loadtxt(trainDataPath,
    dtype=np.float32, delimiter=",")
  print("\nTrain data: ")
  showMatrixPartial(trainDataMatrix, 2, 1, True)
  testDataPath = "irisTestData.txt"
  testDataMatrix = np.loadtxt(testDataPath,
    dtype=np.float32, delimiter=",")
  
  maxEpochs = 50
  learnRate = 0.05
  momentum = 0.75
  print("\nSetting maxEpochs = " + str(maxEpochs))
  print("Setting learning rate = %0.3f " % learnRate)
  print("Setting momentum = %0.3f " % momentum)

  print("\nStarting training without momentum")
  nn.train(trainDataMatrix, maxEpochs, learnRate, 0.0)
  print("Training complete")
  
  accTrain = nn.accuracy(trainDataMatrix)
  accTest = nn.accuracy(testDataMatrix)
  
  print("\nAccuracy on 120-item train data = %0.4f "
    % accTrain)
  print("Accuracy on 30-item test data   = %0.4f "
    % accTest)

  nn = NeuralNetwork(numInput, numHidden, numOutput, seed)
  print("\nStarting training with momentum")
  nn.train(trainDataMatrix, maxEpochs, learnRate, momentum)
  print("Training complete")
  
  accTrain = nn.accuracy(trainDataMatrix)
  accTest = nn.accuracy(testDataMatrix)
  
  print("\nAccuracy on 120-item train data = %0.4f "
    % accTrain)
  print("Accuracy on 30-item test data   = %0.4f "
    % accTest)
  loop maxEpochs times
  for-each training item
    compute output, hidden node signals
    compute weight gradients
    compute weight deltas and save
    update weights using gradients
    update weights using momentum
  end-for
end-loop
def train(self, trainData, maxEpochs, learnRate):
  hoGrads = np.zeros(shape=[self.nh, self.no],
    dtype=np.float32)
  obGrads = np.zeros(shape=[self.no],
    dtype=np.float32)
  ihGrads = np.zeros(shape=[self.ni, self.nh],
    dtype=np.float32)
  hbGrads = np.zeros(shape=[self.nh],
    dtype=np.float32)
    
  print("\nEnd demo \n")
   
if __name__ == "__main__":
  main()
