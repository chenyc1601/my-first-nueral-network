# -*- coding: utf-8 -*-
import numpy as np

# read dataset from file to list
def read_input_data(dataPath) :
    dataFile = open(dataPath, 'r')
    dataList = dataFile.readlines()
    dataFile.close()
    return dataList

# scale inputs from [0, 255] to [0.01, 1.00]
def scale_inputs(dataList) :
    allValues = dataList[0].split(',')
    scaledInput = np.asfarray(allValues[1:] / 255.0 * 0.99) + 0.01
    return scaledInput
