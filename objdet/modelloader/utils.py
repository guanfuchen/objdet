# -*- coding: utf-8 -*-
import torch
import random


def colors(n):
    """
    获得n个随机初始化的rgb颜色
    :param n:
    :return:
    """
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


def change_box_format(boxes, box_format):
    """
    改变box的表示格式，(cx,cy,h,w)<->(x_lt, y_lt, x_rb, y_rb)
    :param boxes:
    :param box_format: either 'xyhw2xyxy' or 'xyxy2xyhw'
    :return:
    """
    assert box_format in ['xywh2xyxy', 'xyxy2xywh']
    if box_format == 'xywh2xyxy':
        cxcy = boxes[:, :2]
        wh = boxes[:, 2:]
        x_lt_y_lt = cxcy-wh/2
        x_rb_y_rb = cxcy+wh/2
        return torch.cat([x_lt_y_lt, x_rb_y_rb], 1)
    elif box_format == 'xyxy2xywh':
        x_lt_y_lt = boxes[:, :2]
        x_rb_y_rb = boxes[:, 2:]
        wh = x_rb_y_rb - x_lt_y_lt
        cxcy = x_lt_y_lt+wh/2
        return torch.cat([cxcy, wh], 1)


def box_iou(box1, box2):
    """
    计算两个box之间的IOU，其中box1为default_box_xyxy(format 为xyxy)，box2为bounding box
    :param box1: default_boxes，[#default_boxes, 4]
    :param box2: bounding_boxes，[#bounding_boxes, 4]
    :return:
    iou，sized [#default_boxes, #bounding_boxes]
    """
    # print('box1.size():{}'.format(box1.size()))
    # print('box2.size():{}'.format(box2.size()))
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [#default_boxes, #bounding_boxes, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [#default_boxes, #bounding_boxes, 2]
    # print('lt:{}'.format(lt))
    # print('rb:{}'.format(rb))

    wh = (rb-lt).clamp(min=0)  # [#default_boxes, #bounding_boxes, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [#default_boxes, #bounding_boxes]
    # print('inter:{}'.format(inter))

    area1 = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])  # [#default_boxes]
    area2 = (box2[:, 2]-box2[:, 0])*(box2[:, 3]-box2[:, 1])  # [#bounding_boxes]
    # print('area1:{}'.format(area1))
    # print('area2:{}'.format(area2))

    iou = inter / (area1[:, None] + area2 - inter)
    # print('iou:{}'.format(iou))

    return iou


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.long)
