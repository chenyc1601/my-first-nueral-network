# -*- coding: utf-8 -*-
import numpy as np

class Dataset :
    """从csv文件中读取数据，并转化成用于训练的内容
    """
    def __init__(self, dataPath) :
        """
        """
        self.dataList = []
        dataFile = open(dataPath, 'r')

        # 数据预处理
        for data in dataFile.readlines() :
            dataValues = data.split(',')
            i = np.asfarray(dataValues[1:]) / 255.0 * 0.99 + 0.01
            t = np.zeros(10) + 0.01
            t[int(dataValues[0])] = 0.99

            self.dataList.append({'input': i, 'target': t})

        dataFile.close()
