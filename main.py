# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork

if __name__ == "__main__" :
    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train_1-100.csv"
    testFilePath = "mnist_dataset/mnist_test_1-10.csv"

    # 输入层，中间层和输出层的节点数
    inputNodes = 28 * 28  # 28*28像素  
    hiddenNodes = 100
    outputNodes = 10  # 0-9十个数字

    # 学习率
    learningRate = 0.5

    # 读取和预处理数据集
    trainSet = Dataset(trainFilePath)

    # 建立并训练网络
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    for data in trainSet.dataList :
        n.train(data.input, data.target)

    # 测试
    for data in testSet.dataList :
        n.test(data.input, data.target)

    # test n.query()
    ## print(n.query([1.0, 0.5, -1.5]))
