# -*- coding: utf-8 -*-

import PIL.Image as pImage
import numpy as np

class Image :
    """将jpg RGB图片处理为用于识别的数据
    """
    def __init__(self, imgPath) :
        img = pImage.open(imgPath)
        imgGray = img.convert('L')  # 读灰度图
        img28 = imgGray.resize((28, 28))  # 转28*28
        array = np.array(img28)  # 转矩阵
        array1_784 = array.reshape((1, 784))  # 转为1*784
        arrayInvert = 255.0 - array1_784  # 白底转黑底
        self.imgData = np.asfarray(arrayInvert) / 255.0 * 0.99 + 0.01
    
if __name__ == "__main__" :
    testImage = Image("test_img/0.jpg")
    print(testImage.imgData)
    