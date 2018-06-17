#!/usr/bin/python
# -*- coding: UTF-8 -*-

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


class TestSSD(unittest.TestCase):

    def test_ssd300(self):
        C, H, W = (3, 300, 300)
        x = Variable(torch.randn(1, C, H, W))
        net = ssd.SSD300(num_classes=21)
        loc_preds, cls_preds = net(x)

    def test_ssd_box_coder(self):
        C, H, W = (3, 300, 300)
        net = ssd.SSD300(num_classes=21)
        ssd_box_coder = ssd.SSDBoxCoder(net)
        ssd_default_boxes = ssd_box_coder.default_boxes
        ssd_default_boxes_xyxy = utils.change_box_format(ssd_default_boxes, 'xywh2xyxy')
        ssd_default_boxes_xyxy_np = ssd_default_boxes_xyxy.numpy()
        # print(ssd_default_boxes)
        # print(ssd_default_boxes_xyxy_np)
        img = np.zeros((H, W, C), dtype=np.uint8)
        colors = utils.colors(ssd_box_coder.fm_num)
        for box_id, ssd_default_box_xyxy_np in enumerate(ssd_default_boxes_xyxy_np):
            pt1_x, pt1_y = ssd_default_box_xyxy_np[:2]
            pt2_x, pt2_y = ssd_default_box_xyxy_np[2:]
            # print('(pt1_x, pt1_y):({}, {})'.format(pt1_x, pt1_y))
            # print('(pt2_x, pt2_y):({}, {})'.format(pt2_x, pt2_y))
            if box_id<38*38*4:
                color = colors[0]
            elif box_id<38*38*4+19*19*6:
                color = colors[1]
            elif box_id<38*38*4+19*19*6+10*10*6:
                color = colors[2]
            elif box_id<38*38*4+19*19*6+10*10*6+5*5*6:
                color = colors[3]
            elif box_id<38*38*4+19*19*6+10*10*6+5*5*6+3*3*4:
                color = colors[4]
            elif box_id<38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4:
                color = colors[5]
            # cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color=color)
            # cv2.imshow('img', img)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     break
        boxes = torch.from_numpy(np.array([(0, 0, 100, 100), (25, 25, 125, 125), (200, 200, 250, 250), (0, 0, 300, 300)], dtype=np.float32))
        labels = torch.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
        loc_targets, cls_targets = ssd_box_coder.encode(boxes, labels)
        # loc_targets_np = loc_targets.numpy()
        # cls_targets_np = cls_targets.numpy()

    def test_box_iou(self):
        box1 = [(0, 0, 100, 100), (25, 25, 125, 125), (200, 200, 250, 250)]
        box2 = [(50, 50, 150, 150)]
        box1 = np.array(box1, dtype=np.float32)
        box2 = np.array(box2, dtype=np.float32)
        box1 = torch.from_numpy(box1)
        box2 = torch.from_numpy(box2)
        ious = utils.box_iou(box1, box2)
        print('ious:{}'.format(ious))

    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
