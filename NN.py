from __future__ import division
import numpy as np
from sklearn import metrics
import pandas as pd
import csv
import sys
import math


class neuralNet:
	# Instance of the class.
	# Takes the input from the runner class for initialization
	def __init__(self, n_input, n_hidden, n_output):
		self.n_input = n_input
		self._weightH = np.random.randn(n_hidden, n_input) * 0.05 
		self._weightO = np.random.randn(n_output, n_hidden) * 0.05
		self._bH = np.random.randn(n_hidden, 1) * 0.05
		self._bO = np.random.randn(n_output, 1) * 0.05
		self._learningRate = .1 #TODO need to look into this
		self._maxEpoch = 10000;
		self._accuracy = 0.001;


	# calculates the sigmoid by using the below formulation
	# 1/(1 + e ^ (-z))
	def sigmoid(self, x):
		return 1/ (1 + np.exp(-x))

	# Calculates the derivative of the sigmoid function
	def derSigmoid(self, x):
		return x*(1-x)


	# Performs feed forward step for training the neural network
	def forwardPropagation(self, input) :
		zH = np.add(np.dot(self._weightH, input), self._bH.T)
		outputH = self.sigmoid(zH)
		zO = np.add(np.dot(self._weightO, outputH.T), self._bO)
		outputO = self.sigmoid(zO)
		return outputO.T, outputH

	# Performs the back propagation
	def backPropagation(self, outputActual, outputHidden, outputExpected, input):
		deltaO = self.derSigmoid(outputActual) * (outputExpected - outputActual)
		dWO = deltaO.T * self._learningRate * outputHidden
		self._weightO = self._weightO + dWO

		deltaH = self.derSigmoid(outputHidden) * (np.dot(self._weightO.T, deltaO.T)).T
		dWH = deltaH.T * self._learningRate * input
		self._weightH = self._weightH + dWH
	
	# Trains the input train dataset
	def train(self, input):
		numEpoch = 0
		errorRed = 100
		numRec = len(input)
		lastError = numRec
		
		while (numEpoch < self._maxEpoch and errorRed > self._accuracy) :
			numEpoch = numEpoch + 10

			for i in xrange(9) :
				for rec in input:
					outputActual, outputH = self.forwardPropagation(rec[:self.n_input])
					self.backPropagation(outputActual, outputH, rec[self.n_input:], rec[:self.n_input])
			
			error = 0
			for rec in input:
				outputActual, outputH = self.forwardPropagation(rec[:self.n_input])
				self.backPropagation(outputActual, outputH, rec[self.n_input:], rec[:self.n_input])			
				if (self._category(outputActual[0]) != self._category(rec[self.n_input:])) :
					error = error + 1
			
			errorRed = abs(lastError - error)/numRec
			#print numEpoch, errorRed, lastError, error, numRec
			lastError = error;

		return lastError

	# Returns the output category with maximum probability
	def _category(self, data) :
		return np.argmax(data) + 1

	# Predicts the output category for the input
	def predict(self, input) :
		output , outputH = self.forwardPropagation(input)
		return self._category(output)








		
