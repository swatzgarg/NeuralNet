from __future__ import division
import numpy as np
import sys
import NN

from sklearn.cross_validation import train_test_split

# This is the runner script that reads the csv file and runs the neural network

# Reads the input file and converts into the array
def readRecords(infile) :
	finalinput = []
	for line in infile :
		finalinput.append(line.split())
	return np.array(finalinput, dtype=np.float64)


trainingdatafile = open('newdata.txt', 'r')

inputdata= readRecords(trainingdatafile)
datasize = len(inputdata)

# Randomly splits the file in 80-20 split for test and train data
trainingbatch , testbatch = train_test_split(inputdata, test_size=0.2)

# Values for the no of hidden, input and output neurons
numHiddenCells = 64
numInputCells = 9
numOutputCells = len(inputdata[0]) - numInputCells


nn = NN.neuralNet(numInputCells, numHiddenCells, numOutputCells)

errTrain = nn.train(trainingbatch)
errPred = 0

for test in testbatch:
	output = nn.predict(test[:numInputCells])
	#print output, np.argmax(test[numInputCells:]) + 1
	if (output != np.argmax(test[numInputCells:]) + 1) :
		errPred =  errPred + 1

print errTrain, len(trainingbatch)
print errPred, len(testbatch)





          