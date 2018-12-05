# -*- coding: utf-8 -*-

import numpy as np
import csv

class Dataset :
    """从csv文件中读取数据，并转化成用于训练的内容
    """
    def __init__(self, dataPath) :
        """
        """
        self.dataList = []

        # 数据预处理
        with open(dataPath, newline='') as csvfile :
            reader = csv.reader(csvfile)
            for row in reader :
                i = np.asfarray(row[1:]) / 255.0 * 0.99 + 0.01
                t = np.zeros(10) + 0.01
                t[int(row[0])] = 0.99
                self.dataList.append({'input': i, 'target': t})

if __name__ == "__main__" :
    dataset = Dataset("mnist_dataset/mnist_test_1-10.csv")
    print(dataset.dataList)
