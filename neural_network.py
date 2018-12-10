# -*- coding: utf-8 -*-

import numpy as np
from scipy import special

# neural network class definition
class NeuralNetwork :

    def __init__(self, learningrate) :
        """根据输入的参数初始化神经网络
        """
        # 各层结点数
        self.inodes = 28 * 28  # 28*28像素
        self.h1nodes = 250  # 第一层
        self.h2nodes = 50  # 第二层
        self.onodes = 10  # 0-9十个数字

        # 学习率
        self.lr = learningrate

        # 链接权重矩阵
        # 权重w_i_j是上层结点i到下层结点j的权重，即：
        # w_1_1 w_2_1
        # w_1_2 w_2_2 ...
        ## weights randomly range from -0.5 to +0.5
        ## self.w_I_H_r = np.random.rand(self.hnodes, self.inodes) - 0.5  
        ## self.w_H_O_r = np.random.rand(self.onodes, self.hnodes) - 0.5
        # weights 正态分布，均值为0，标准差为1/传入链接数目的开方
        self.w1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.h1nodes, self.inodes))
        self.w2 = np.random.normal(0.0, pow(self.h1nodes, -0.5), (self.h2nodes, self.h1nodes))
        self.w3 = np.random.normal(0.0, pow(self.h2nodes, -0.5), (self.onodes, self.h2nodes))

        # 使用sigmoid函数作为结点的激发函数
        self.activation_function = lambda x: special.expit(x)

        return


    def query(self, aInput) :
        """输入一个1*784矩阵，返回10*1矩阵<ndarray>
        """
        # convert inputs list to 2d array
        inputs = np.array(aInput, ndmin=2).T

        h1Inputs = np.dot(self.w1, inputs)
        h1Outputs = self.activation_function(h1Inputs)

        h2Inputs = np.dot(self.w2, h1Outputs)
        h2Outputs = self.activation_function(h2Inputs)

        finalInputs = np.dot(self.w3, h2Outputs)
        finalOutputs = self.activation_function(finalInputs)

        return finalOutputs


    def train(self, aInput, aTarget) :
        """输入一个数据，对神经网络进行一次训练
        """
        # convert inputs list to 2d array
        inputs = np.array(aInput, ndmin=2).T

        h1Inputs = np.dot(self.w1, inputs)
        h1Outputs = self.activation_function(h1Inputs)

        h2Inputs = np.dot(self.w2, h1Outputs)
        h2Outputs = self.activation_function(h2Inputs)

        finalInputs = np.dot(self.w3, h2Outputs)
        finalOutputs = self.activation_function(finalInputs)

        # 计算误差
        targets = np.array(aTarget, ndmin=2).T
        outputErrors = targets - finalOutputs
        h2Errors = np.dot(self.w3.T, outputErrors)
        h1Errors = np.dot(self.w2.T, h2Errors)

        # 调整权重
        self.w3 += self.lr * np.dot(outputErrors * finalOutputs * (1.0 - finalOutputs), h2Outputs.T)
        self.w2 += self.lr * np.dot(h2Errors * h2Outputs * (1.0 - h2Outputs), h1Outputs.T)
        self.w1 += self.lr * np.dot(h1Errors * h1Outputs * (1.0 - h1Outputs), inputs.T)

        return


    def test(self, inArray, tarArray) :
        """输入输入矩阵和目标矩阵，返回计算结果和目标结果<int>
        """
        ouDigi = trans(self.query(inArray))
        tarDigi = trans(tarArray)
        return [ouDigi, tarDigi]


    def guess(self, inArray) :
        """输入1*784输入矩阵，返回结果数字
        """
        ouArray = self.query(inArray)
        ## print(ouArray)  ## 查看结果
        ouDigi= trans(ouArray)
        return ouDigi


def trans(ouArray) :
    """将10*1结果矩阵转为对应的数字
    """
    ouList = ouArray.tolist()
    ouDigi = ouList.index(max(ouList))
    return ouDigi
