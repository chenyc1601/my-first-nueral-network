# -*- coding: utf-8 -*-

import numpy as np
from scipy import special

# neural network class definition
class NeuralNetwork :

    #initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        """
        """
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # link weight matrices, w_I_H and w_H_O
        # weight inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w_1_1 w_2_1
        # w_1_2 w_2_2 etc
        # weights randomly range from -0.5 to +0.5
        ## self.w_I_H_r = np.random.rand(self.hnodes, self.inodes) - 0.5  
        ## self.w_H_O_r = np.random.rand(self.onodes, self.hnodes) - 0.5
        ## weights 正态分布，均值为0，标准差为1/传入链接数目的开方
        self.w_I_H_n = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.w_H_O_n = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x: special.expit(x)

        return

    # train the neural network
    def train(self, inputsList, targetsList) :
        # convert lists to 2d arrays
        inputs = np.array(inputsList, ndmin=2).T
        targets = np.array(targetsList, ndmin=2).T

        hiddenInputs = np.dot(self.w_I_H_n, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)

        finalInputs = np.dot(self.w_H_O_n, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        # calculate errors
        outputErrors = targets - finalOutputs
        hiddenErrors = np.dot(self.w_H_O_n.T, outputErrors)

        # update the weights for the links between layers
        self.w_H_O_n += self.lr * np.dot(outputErrors * finalOutputs * (1.0 - finalOutputs), hiddenOutputs.T)
        self.w_I_H_n += self.lr * np.dot(hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs), inputs.T)

        return

    # query the neural network
    def query(self, inputsList) :
        # convert inputs list to 2d array
        inputs = np.array(inputsList, ndmin=2).T

        hiddenInputs = np.dot(self.w_I_H_n, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)

        finalInputs = np.dot(self.w_H_O_n, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        return finalOutputs
