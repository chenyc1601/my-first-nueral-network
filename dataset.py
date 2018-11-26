# -*- coding: utf-8 -*-
import numpy as np

class Dataset :
    """从csv文件中读取数据，并转化成用于训练的内容
    """
    dataList = []
    def __init__(self, dataPath) :
        """
        """
        # 从文件读取元数据
        dataFile = open(dataPath, 'r')
        metaDataList = dataFile.readlines()
        dataFile.close()

        # 数据预处理
        for data in metaDataList :
            dataValues = data.split(',')
            i = np.asfarray(dataValues[1:] / 255.0 * 0.99) + 0.01
            t = np.zeros(10) + 0.01
            t[int(dataValues[0])] = 0.99

            self.dataList.append({'input': i, 'target': t})
