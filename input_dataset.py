# -*- coding: utf-8 -*-
import numpy as np

class Dataset :
    """从csv文件中读取数据，转化成用于训练的内容
    """

    def __init__(self, dataPath) :
        """输入数据集文件路径
        csv中的一行是一个数据
        """
        self.inputList = []
        self.targetList = []

        dataFile = open(dataPath, 'r')
        self.dataList = dataFile.readlines()
        dataFile.close()


    def init_inputs(self) :
        """从数据list中读取输入部分，并转化至神经网络需要的数字范围，输出整理后的list
        每个数据从第二位开始是输入部分
        原始数据范围[0, 255]（像素点颜色），转化至[0.01, 1.00]
        """
        for data in self.dataList :
            dataValues = data.split(',')
            scaledInput = np.asfarray(dataValues[1:] / 255.0 * 0.99) + 0.01
            self.inputList.append(scaledInput)

        return self.inputList


    def init_targets(self) :
        """从数据list中读取目标部分，并转化至神经网络需要的数字范围，输出整理后的list
        每个数据第一位是目标数字
        转化为十位的目标输出（0-9），范围是[0.01, 0.99]
        """
        for data in self.dataList :
            dataValues = data.split(',')
            target = np.zeros(10) + 0.01
            target[int(dataValues[0])] = 0.99
            self.targetList.append(target)

        return self.targetList
