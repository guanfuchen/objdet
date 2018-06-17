# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import math

from . import utils


class StrideConv(nn.Module):
    """
    StrideConv：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param groups:
        :param bias:
        """
        super(StrideConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class StridePool(nn.Module):
    """
    StridePool：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """

    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super(StridePool, self).__init__()
        padding = (kernel_size - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        return self.pool(x)


class L2Norm(nn.Module):
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x


class SSDBoxCoder:
    def __init__(self, ssd_model):
        """
        :type ssd_model: SSD300
        """
        self.steps = ssd_model.steps
        self.fm_sizes = ssd_model.fm_sizes
        self.fm_num = len(self.fm_sizes)
        self.aspect_ratios = ssd_model.aspect_ratios
        self.box_sizes = ssd_model.box_sizes
        self.default_boxes = self._get_default_boxes()
        self.variances = (0.1, 0.2)

    def _get_default_boxes(self):
        """
        :return: boxes: (#boxes, 4), 4 is for (cx, cy, h, w) box format
        """
        boxes = []
        for fm_id, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                # print('(h,w):({},{})'.format(h, w))
                cx = (w + 0.5) * self.steps[fm_id]  # steps recover the center to the origin map
                cy = (h + 0.5) * self.steps[fm_id]  # steps recover the center to the origin map
                # print('(cx,cy):({},{})'.format(cx, cy))

                s = self.box_sizes[fm_id]
                boxes.append((cx, cy, s, s))  # boxes append (cx, cy, h, w)

                s_prime = math.sqrt(self.box_sizes[fm_id] * self.box_sizes[fm_id + 1])  # append large box
                boxes.append((cx, cy, s_prime, s_prime))  # boxes append (cx, cy, h, w)

                # aspect_ratio just save 2, 3 and append 1/2, 1/3
                for aspect_ratio in self.aspect_ratios[fm_id]:
                    boxes.append((cx, cy, s / math.sqrt(aspect_ratio),
                                  s * math.sqrt(aspect_ratio)))  # boxes append (cx, cy, h, w)
                    boxes.append((cx, cy, s * math.sqrt(aspect_ratio),
                                  s / math.sqrt(aspect_ratio)))  # boxes append (cx, cy, h, w)

        return torch.Tensor(boxes)

    def encode(self, boxes, labels):
        """
        SSD编码规则：
            tx = (x-anchor_x) / (variance[0]*anchor_w)
            ty = (y-anchor_y) / (variance[0]*anchor_h)
            tw = log(w/anchor_w) / variance[1]
            th = log(h/anchor_h) / variance[1]
        :param boxes: 输入的bounding boxes格式为（x_lt, y_lt, x_rb, y_rb），size [#obj, 4]
        :param labels:输入的目标类的标签，size [#obj, ]
        :return:
        """

        def argmax(x):
            x_v, x_i = x.max(0)
            x_j = x_v.max(0)[1][0]
            return x_i[x_j], x_j

        default_boxes = self.default_boxes  # xywh
        default_boxes_xyxy = utils.change_box_format(default_boxes, 'xywh2xyxy')

        ious = utils.box_iou(default_boxes_xyxy, boxes)  # 计算boxes和默认的boxes之间的IOU
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        # 不断寻找到最大值，直到最大值也比较小的时候
        while True:
            i, j = argmax(masked_ious)
            # print('(i,j):({},{})'.format(i, j))
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0
        # masked_ious_np = masked_ious.numpy()
        # ious_np = ious.numpy()
        # index_np = index.numpy()
        # print(masked_ious)

        mask = (index < 0) & (ious.max(1)[0] >= 0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]
        boxes_xywh = utils.change_box_format(boxes, 'xyxy2xywh')

        # ssd tx ty tw th编码
        loc_xy = (boxes_xywh[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / self.variances[0]
        loc_wh = torch.log(boxes_xywh[:, 2:] / default_boxes[:, 2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        xy = loc_preds[:, :2] * self.variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * self.variances[1]) * self.default_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = utils.box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores


class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()
        self.conv1_1 = StrideConv(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv1_2 = StrideConv(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = StridePool(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1 = StrideConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_2 = StrideConv(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = StridePool(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1 = StrideConv(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv3_2 = StrideConv(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv3_3 = StrideConv(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool3 = StridePool(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = StrideConv(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv4_2 = StrideConv(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv4_3 = StrideConv(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.norm4 = L2Norm(512, 20)  # 使用Norm层正则化
        self.pool4 = StridePool(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5_1 = StrideConv(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5_2 = StrideConv(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5_3 = StrideConv(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool5 = StridePool(kernel_size=3, stride=1, ceil_mode=True)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)  # 这个conv特别需要注意
        self.conv7 = StrideConv(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)

        self.conv8_1 = StrideConv(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
        self.conv8_2 = StrideConv(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.conv9_1 = StrideConv(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        self.conv9_2 = StrideConv(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.conv10_1 = StrideConv(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

        self.conv11_1 = StrideConv(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

    def forward(self, x):
        xs = []
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x1 = self.norm4(x)
        # print('x1.size():{}'.format(x1.size()))
        xs.append(x1)  # conv4_3 38*38*512
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = F.relu(x)
        x2 = x
        # print('x2.size():{}'.format(x2.size()))
        xs.append(x2)  # conv7 19*19*1024

        x = self.conv8_1(x)
        x = F.relu(x)
        x = self.conv8_2(x)
        x = F.relu(x)
        x3 = x
        # print('x3.size():{}'.format(x3.size()))
        xs.append(x3)  # conv8_2 10*10*512

        x = self.conv9_1(x)
        x = F.relu(x)
        x = self.conv9_2(x)
        x = F.relu(x)
        x4 = x
        # print('x4.size():{}'.format(x4.size()))
        xs.append(x4)  # conv9_2 5*5*256

        x = self.conv10_1(x)
        x = F.relu(x)
        x = self.conv10_2(x)
        x = F.relu(x)
        x5 = x
        # print('x5.size():{}'.format(x5.size()))
        xs.append(x5)  # conv10_2 3*3*256

        x = self.conv11_1(x)
        x = F.relu(x)
        x = self.conv11_2(x)
        x = F.relu(x)
        x6 = x
        # print('x6.size():{}'.format(x6.size()))
        xs.append(x6)  # conv11_2 1*1*256

        # print('x.size():{}'.format(x.size()))
        return xs


class SSD300(nn.Module):
    steps = (8, 16, 32, 64, 100, 300)  # steps for recover to the origin image size
    fm_sizes = (38, 19, 10, 5, 3, 1)  # feature map size
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))  # aspect ratio
    box_sizes = (30, 60, 111, 162, 213, 264, 315)  # box size

    def __init__(self, num_classes=21):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        self.extractor = VGG16Extractor300()

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []

        xs = self.extractor(x)

        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


class SSDLoss(nn.Module):
    def __init__(self, num_classes):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)  # [N,#anchors]

        num_neg = 3 * pos.sum(1)  # [N,]
        neg = rank < num_neg[:, None]  # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        """
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        # ===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        # ===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        # ===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        # ===============================================================
        cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets < 0] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos | neg].sum()

        print('loc_loss: {} | cls_loss: {}'.format(loc_loss.item() / num_pos, cls_loss.item() / num_pos))
        loss = (loc_loss + cls_loss) / num_pos
        return loss
