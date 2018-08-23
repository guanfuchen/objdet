# -*- coding: utf-8 -*-
import argparse

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import fire

from objdet.modelloader import faster_rcnn
from objdet.utils.config import faster_rcnn_config


def train(args):
    num_classses = 21
    faster_rcnn_config.parse(args)
    net = faster_rcnn.FasterRCNNVGG16(num_classses=num_classses)
    trainer = faster_rcnn.FasterRCNNTrainer(net)
    C, H, W = (3, 300, 300)
    x = Variable(torch.randn(1, C, H, W))
    boxes = np.array([(0, 0, 100, 100), (0, 0, 100, 100), (0, 0, 100, 100)])[np.newaxis, :]
    boxes = torch.from_numpy(boxes)
    labels = np.array([1, 1, 1], dtype=np.long)[np.newaxis, :]
    labels = torch.from_numpy(labels)
    scale = 1.0

    print("boxes.shape:", boxes.shape)
    print("labels.shape:", labels.shape)

    # optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    # criterion = ssd.SSDLoss(num_classes=num_classses)
    #
    for epoch in range(faster_rcnn_config.epoch):
        pass
        trainer.train_step(x, boxes, labels, scale)
    #     loc_preds, cls_preds = net(x)
    #     # print('loc_preds.size():{}'.format(loc_preds.size()))
    #     # print('cls_preds.size():{}'.format(cls_preds.size()))
    #     optimizer.zero_grad()
    #
    #     loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
    #     loss.backward()
    #     optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training faster rcnn parameter setting')
    parser.add_argument('--epoch', type=int, default=1, help='train epoch [ 1 ]')
    parser.add_argument('--lr', type=float, default=0.0001, help='train learning rate [ 0.0001 ]')
    # parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    # parser.addargument('--validate_model', type=str, default='', help='validate model path [ fcn32s_camvid_9.pkl ]')
    # parser.add_argument('--validate_model_state_dict', type=str, default='', help='validate model state dict path [ fcn32s_camvid_9.pt ]')
    # parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    # parser.add_argument('--n_classes', type=int, default=13, help='train class num [ 13 ]')
    # parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    # parser.add_argument('--blend', type=bool, default=False, help='blend the result and the origin [ False ]')
    args = parser.parse_args()

    train(args)
