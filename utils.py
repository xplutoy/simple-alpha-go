# -*- coding: utf-8 -*-
# @Time  : 2019/3/12 20:18
# @Author : yx
# @Desc : ==============================================

import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
