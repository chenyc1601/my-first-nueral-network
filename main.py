# -*- coding: utf-8 -*-

from neural-network import NeuralNetwork

if __name__ == "__main__" :
    # 

    # number of input, hidden and output nodes
    inputNodes = 28 * 28  # 28*28 pixels  
    hiddenNodes = 3
    outputNodes = 10  # 10 digits

    # learning rate is 0.5
    learningRate = 0.5

    # create instance of neural network
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    # test n.query()
    ## print(n.query([1.0, 0.5, -1.5]))

    pass
