# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pickle as pk
import argparse
from img import Image

if __name__ == "__main__" :
    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train.csv"
    testFilePath = "mnist_dataset/mnist_test.csv"

    # 定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgFile", type=str, help="待识别的图片文件", default="test_img/0.png")
    parser.add_argument("--wIH", type=str, help="输入-隐藏权重矩阵文件")
    parser.add_argument("--wHO", type=str, help="隐藏-输出权重矩阵文件")
    args = parser.parse_args()

    # 输入层，中间层和输出层的节点数
    inputNodes = 28 * 28  # 28*28像素  
    hiddenNodes = 200
    outputNodes = 10  # 0-9十个数字

    # 学习率
    learningRate = 0.1

    # 初始化
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    print("网络初始化完成")

    # 读取或训练网络
    if args.wIH and args.wHO :  # 读取
        with open(args.wIH, 'rb') as handle :
            n.w_I_H_n = pk.load(handle)
        with open(args.wHO, 'rb') as handle :
            n.w_H_O_n = pk.load(handle)
    else :  # 训练
        trainSet = Dataset(trainFilePath)
        for data in trainSet.dataList :
            n.train(data['input'], data['target'])
        print("训练完成")

        with open('w_I_H.pickle', 'wb') as handle :
            pk.dump(n.w_I_H_n, handle, protocol=pk.HIGHEST_PROTOCOL)
        with open('w_H_O.pickle', 'wb') as handle :
            pk.dump(n.w_H_O_n, handle, protocol=pk.HIGHEST_PROTOCOL)        

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
    testImg = Image(args.imgFile)
    digi = n.guess(testImg.gray784)
    print(digi)
