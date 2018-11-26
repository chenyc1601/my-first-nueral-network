# -*- coding: utf-8 -*-

import numpy as np
from scipy import special

# neural network class definition
class NeuralNetwork :

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        """根据输入的参数初始化神经网络
        """
        # 各层结点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

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
        self.w_I_H_n = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.w_H_O_n = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 使用sigmoid函数作为结点的激发函数
        self.activation_function = lambda x: special.expit(x)

        return


    def query(self, aInput) :
        """输入一个数据，返回最终输出矩阵<ndarray>
        """
        # convert inputs list to 2d array
        inputs = np.array(aInput, ndmin=2).T

        hiddenInputs = np.dot(self.w_I_H_n, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)

        finalInputs = np.dot(self.w_H_O_n, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        return finalOutputs


    def train(self, aInput, aTarget) :
        """输入一个数据，对神经网络进行一次训练
        """
        # convert inputs list to 2d array
        inputs = np.array(aInput, ndmin=2).T

        hiddenInputs = np.dot(self.w_I_H_n, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)

        finalInputs = np.dot(self.w_H_O_n, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        # 计算误差
        targets = np.array(aTarget, ndmin=2).T
        outputErrors = targets - finalOutputs
        hiddenErrors = np.dot(self.w_H_O_n.T, outputErrors)

        # 调整权重
        self.w_H_O_n += self.lr * np.dot(outputErrors * finalOutputs * (1.0 - finalOutputs), hiddenOutputs.T)
        self.w_I_H_n += self.lr * np.dot(hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs), inputs.T)

        return


    def test(self, aInput, aTarget) :
        """输入一个数据，返回计算结果和目标结果<int>
        """
        finalOutputList = self.query(aInput).tolist()
        ouDigi = finalOutputList.index(max(finalOutputList))
        targetList = aTarget.tolist()
        tarDigi = targetList.index(max(targetList))
        
        return [ouDigi, tarDigi]
