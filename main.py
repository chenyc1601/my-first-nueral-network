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
    parser.add_argument("-t", "--train", help="重新训练网络", action="store_true")
    parser.add_argument("-e", "--echo", type=int, help="训练次数/世代", default=5)
    parser.add_argument("-r", "--rate", type=float, help="学习率", default=0.1)
    args = parser.parse_args()

    # 数据集路径
    trainFilePath = "mnist_dataset/mnist_train.csv"  # 训练集
    testFilePath = "mnist_dataset/mnist_test.csv"  # 测试集
    weight1 = "w1.pickle"
    weight2 = "w2.pickle"
    weight3 = "w3.pickle"

    # 学习率
    learningRate = args.rate

    # 初始化
    print("网络初始化开始...")
    n = NeuralNetwork(learningRate)
    print("网络初始化完成！")

    # 读取或训练网络
    if args.train != True :  # 读取
        with open(weight1, 'rb') as handle :
            n.w1 = pk.load(handle)
        with open(weight2, 'rb') as handle :
            n.w2 = pk.load(handle)
        with open(weight3, 'rb') as handle :
            n.w3 = pk.load(handle)
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
        with open('w1.pickle', 'wb') as handle :
            pk.dump(n.w1, handle, protocol=pk.HIGHEST_PROTOCOL)
        with open('w2.pickle', 'wb') as handle :
            pk.dump(n.w2, handle, protocol=pk.HIGHEST_PROTOCOL)        
        with open('w3.pickle', 'wb') as handle :
            pk.dump(n.w3, handle, protocol=pk.HIGHEST_PROTOCOL)        

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
    ## testImg = Image(args.imgFile)
    for i in range(10) :
        testImg = Image("test_img/{0}.jpg".format(i))
        digi = n.guess(testImg.imgData)
        print("图{0}中数字为：{1}".format(i, digi))
