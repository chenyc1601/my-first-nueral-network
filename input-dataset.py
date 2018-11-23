# -*- coding: utf-8 -*-
import numpy as np

# read dataset from file to list
dataFile = open("./mnist_dataset/mnist_train_1-100.csv", 'r')
dataList = dataFile.readlines()
dataFile.close()

# scale inputs from [0, 255] to [0.01, 1.00]
allValues = dataList[0].split(',')
scaledInput = np.asfarray(allValues[1:] / 255.0 * 0.99) + 0.01
