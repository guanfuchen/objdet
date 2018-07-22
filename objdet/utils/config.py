# -*- coding: utf-8 -*-
import os


class FasterRCNNConfig(object):
    def __init__(self):
        self.voc_data_dir = os.path.expanduser('~/Data/VOC/VOCdevkit/VOC2007')
        self.epoch = 1
        self.lr = 0.1

    def parse(self, args):
        # print(args.__dict__)
        state_dict = self.state_dict()
        # print('state_dict:', state_dict)
        for k, v in args.__dict__.items():
            # print('k:{},v:{}'.format(k, v))
            if k in state_dict:
                setattr(self, k, v)

        print('state_dict:', self.state_dict())

    def state_dict(self):
        state_dict = {k: getattr(self, k) for k, _ in self.__dict__.items()}
        return state_dict


faster_rcnn_config = FasterRCNNConfig()
