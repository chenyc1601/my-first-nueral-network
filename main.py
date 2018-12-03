# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pickle as pk
import argparse

if __name__ == "__main__" :
    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train.csv"
    testFilePath = "mnist_dataset/mnist_test.csv"
    trainedNetworkPath = ""

    """ # 定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument("imgFile", type=str, help="待识别的图片文件")
    parser.add_argument("--demo", )
    args = parser.parse_args() """

    # 输入层，中间层和输出层的节点数
    inputNodes = 28 * 28  # 28*28像素  
    hiddenNodes = 200
    outputNodes = 10  # 0-9十个数字

    # 学习率
    learningRate = 0.1

    # 读取或训练网络
    if trainedNetworkPath :  # 读取
        pass
    else :  # 训练
        n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
        print("网络初始化完成")

        trainSet = Dataset(trainFilePath)
        for data in trainSet.dataList :
            n.train(data['input'], data['target'])
        print("训练完成")

    # 测试
    testSet = Dataset(testFilePath)
    errorCount = 0
    totalCount = 0
    for data in testSet.dataList :
        totalCount += 1
        result = n.test(data['input'], data['target'])
        if result[0] != result[1] :  # 结果错误
            ## print(result)  # 显示结果
            errorCount += 1  # 记录错误
            ## imageArray = data['input'].reshape((28, 28))  # 显示错误图像
            ## plt.imshow(imageArray, cmap='Greys', interpolation='None')
            ## plt.draw()
            ## plt.pause(5)
    print(float(1 - errorCount / totalCount))

    # 实际识别
    
