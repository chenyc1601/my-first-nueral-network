# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork
import matplotlib.pyplot

if __name__ == "__main__" :
    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train.csv"
    testFilePath = "mnist_dataset/mnist_test.csv"

    # 输入层，中间层和输出层的节点数
    inputNodes = 28 * 28  # 28*28像素  
    hiddenNodes = 200
    outputNodes = 10  # 0-9十个数字

    # 学习率
    learningRate = 0.1

    # 读取和预处理数据集
    trainSet = Dataset(trainFilePath)
    testSet = Dataset(testFilePath)
    print("finish reading data")

    # 建立并训练网络
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    print("finish initiating nn")
    for data in trainSet.dataList :
        n.train(data['input'], data['target'])
    print("finish training")

    # 测试
    errorCount = 0
    totalCount = 0
    for data in testSet.dataList :
        totalCount += 1
        result = n.test(data['input'], data['target'])
        if result[0] != result[1] :  # 结果错误
            ## print(result)  # 显示结果
            errorCount += 1  # 记录错误
            ## imageArray = data['input'].reshape((28, 28))  # 显示错误图像
            ## matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None').make_image()
    print(float(1 - errorCount / totalCount))

    # test n.query()
    ## print(n.query([1.0, 0.5, -1.5]))
