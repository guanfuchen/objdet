# -*- coding: utf-8 -*-
import unittest
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import cv2

from context import objdet
from objdet.modelloader import ssd, utils


class TestFasterRCNN(unittest.TestCase):

    def test_roi_pooling(self):
        C, H, W = (6, 300, 300)
        x = Variable(torch.randn(2, C, H, W))
        rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        out = utils.roi_pooling(x, rois, size=(3, 3))
        print('out.size():', out.size())

    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
