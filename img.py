# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import PIL.Image as pImage
import numpy as np

class Image :
    """将图片处理为用于识别的数据
    """
    def __init__(self, imgPath) :
        self.img = pImage.open(imgPath).convert('L')
        self.array = np.array(self.img)
        # 转为1*784
        self.gray784 = self.array.reshape((1, 784))
    
if __name__ == "__main__" :
    img = Image("test_img/1.jpg")
    img.img.save("test_img/xx.png")
