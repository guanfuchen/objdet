# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

from objdet.modelloader import ssd


def train():
    num_classses = 21
    net = ssd.SSD300(num_classes=num_classses)
    ssd_box_coder = ssd.SSDBoxCoder(net)

    C, H, W = (3, 300, 300)
    x = Variable(torch.randn(1, C, H, W))
    boxes = torch.from_numpy(np.array([(0, 0, 100, 100), (25, 25, 125, 125), (200, 200, 250, 250), (0, 0, 300, 300)], dtype=np.float32))
    labels = torch.from_numpy(np.array([1, 2, 3, 4], dtype=np.long))
    loc_targets, cls_targets = ssd_box_coder.encode(boxes, labels)
    loc_targets = loc_targets[None, :]
    cls_targets = cls_targets[None, :]
    # print('loc_targets.size():{}'.format(loc_targets.size()))
    # print('cls_targets.size():{}'.format(cls_targets.size()))

    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    criterion = ssd.SSDLoss(num_classes=num_classses)

    for epoch in range(100):
        loc_preds, cls_preds = net(x)
        # print('loc_preds.size():{}'.format(loc_preds.size()))
        # print('cls_preds.size():{}'.format(cls_preds.size()))
        optimizer.zero_grad()

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    train()
