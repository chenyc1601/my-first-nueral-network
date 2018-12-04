# -*- coding: utf-8 -*-

from dataset import Dataset
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pickle as pk
import argparse
from img import Image

if __name__ == "__main__" :

    # 定义输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imgFile", help="待识别的图片文件", default="test_img/0.jpg")
    parser.add_argument("-w", "--weight", help="使用预存网络权重", action="store_true")
    parser.add_argument("-e", "--echo", type=int, help="训练次数/世代", default=5)
    parser.add_argument("-n", "--node", type=int, help="中间层结点数", default=200)
    parser.add_argument("-r", "--rate", type=float, help="学习率", default=0.1)
    args = parser.parse_args()

    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train.csv"  # 训练集
    testFilePath = "mnist_dataset/mnist_test.csv"  # 测试集
    weightInHid = "w_I_H.pickle"
    weightHidOut = "w_H_O.pickle"

    # 网络输入层、中间层和输出层的节点数
    inputNodes = 28 * 28  # 28*28像素  
    hiddenNodes = args.node
    outputNodes = 10  # 0-9十个数字

    # 学习率
    learningRate = args.rate

    # 初始化
    print("网络初始化开始...")
    n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    print("网络初始化完成！")

    # 读取或训练网络
    if args.weight :  # 读取
        with open(weightInHid, 'rb') as handle :
            n.w_I_H_n = pk.load(handle)
        with open(weightHidOut, 'rb') as handle :
            n.w_H_O_n = pk.load(handle)
    else :  # 训练
        print("读取训练集...")
        trainSet = Dataset(trainFilePath)
        print("训练集读取完毕！")
        for e in range(args.echo) :
            print("第{0}次训练开始...".format(e + 1))
            for data in trainSet.dataList :
                n.train(data['input'], data['target'])
            print("第{0}次训练完成！".format(e + 1))
        # 保存训练结果
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
    print("测试集识别正确率：{0}".format(float(1 - errorCount / totalCount)))

    # 实际识别
    testImg = Image(args.imgFile)
    digi = n.guess(testImg.array)
    print("图中数字为：{0}".format(digi))
