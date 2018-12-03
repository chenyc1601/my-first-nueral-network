# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np

class Image :
    """将图片处理为用于识别的数据
    """
    def __init__(self, imgPath) :
        # 读取图片
        self.img = mpimg.imread(imgPath)  # <ndarray>

        # RGB转灰度
        self.gray = np.dot(self.img[..., :3], [0.2989, 0.5870, 0.1140])  # 按维基上的换算公式处理

        # 转为1*784
        self.gray784 = self.gray.reshape((1, 784))
    