# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork

if __name__ == "__main__" :
    # dataset path
    dataPath = "mnist_dataset/mnist_train_1-100.csv"

    # number of input, hidden and output nodes
    inputNodes = 28 * 28  # 28*28 pixels  
    hiddenNodes = 100
    outputNodes = 10  # 10 digits

    # learning rate is 0.5
    learningRate = 0.5

    # read dataset
    data = Dataset(dataPath)
    inputList = data.read_inputs()
    targetList = data.read_targets()

    # create and train neural network
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    for i in inputList :

    # test n.query()
    ## print(n.query([1.0, 0.5, -1.5]))
