# -*- coding: utf-8 -*-
from collections import namedtuple

from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import six
import time

class FasterRCNNBoxCoder:
    def __init__(self):
        pass

    def loc2bbox(self, src_bbox, loc):
        """Decode bounding boxes from bounding box offsets and scales.

        Given bounding box offsets and scales computed by
        :meth:`bbox2loc`, this function decodes the representation to
        coordinates in 2D image coordinates.

        Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
        box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
        the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
        and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
        by the following formulas.

        * :math:`\\hat{g}_y = p_h t_y + p_y`
        * :math:`\\hat{g}_x = p_w t_x + p_x`
        * :math:`\\hat{g}_h = p_h \\exp(t_h)`
        * :math:`\\hat{g}_w = p_w \\exp(t_w)`

        The decoding formulas are used in works such as R-CNN [#]_.

        The output is same type as the type of the inputs.

        .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
        Rich feature hierarchies for accurate object detection and semantic \
        segmentation. CVPR 2014.

        Args:
            src_bbox (array): A coordinates of bounding boxes.
                Its shape is :math:`(R, 4)`. These coordinates are
                :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            loc (array): An array with offsets and scales.
                The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
                This contains values :math:`t_y, t_x, t_h, t_w`.

        Returns:
            array:
            Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
            The second axis contains four values \
            :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
            \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

        """

        if src_bbox.shape[0] == 0:
            return np.zeros((0, 4), dtype=loc.dtype)

        src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

        src_height = src_bbox[:, 2] - src_bbox[:, 0]
        src_width = src_bbox[:, 3] - src_bbox[:, 1]
        src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
        src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

        dy = loc[:, 0::4]
        dx = loc[:, 1::4]
        dh = loc[:, 2::4]
        dw = loc[:, 3::4]

        ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
        ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
        h = np.exp(dh) * src_height[:, np.newaxis]
        w = np.exp(dw) * src_width[:, np.newaxis]

        dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
        dst_bbox[:, 0::4] = ctr_y - 0.5 * h
        dst_bbox[:, 1::4] = ctr_x - 0.5 * w
        dst_bbox[:, 2::4] = ctr_y + 0.5 * h
        dst_bbox[:, 3::4] = ctr_x + 0.5 * w

        return dst_bbox

    def bbox2loc(self, src_bbox, dst_bbox):
        """Encodes the source and the destination bounding boxes to "loc".

        Given bounding boxes, this function computes offsets and scales
        to match the source bounding boxes to the target bounding boxes.
        Mathematcially, given a bounding box whose center is
        :math:`(y, x) = p_y, p_x` and
        size :math:`p_h, p_w` and the target bounding box whose center is
        :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
        :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

        * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
        * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
        * :math:`t_h = \\log(\\frac{g_h} {p_h})`
        * :math:`t_w = \\log(\\frac{g_w} {p_w})`

        The output is same type as the type of the inputs.
        The encoding formulas are used in works such as R-CNN [#]_.

        .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
        Rich feature hierarchies for accurate object detection and semantic \
        segmentation. CVPR 2014.

        Args:
            src_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
                These coordinates are
                :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            dst_bbox (array): An image coordinate array whose shape is
                :math:`(R, 4)`.
                These coordinates are
                :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

        Returns:
            array:
            Bounding box offsets and scales from :obj:`src_bbox` \
            to :obj:`dst_bbox`. \
            This has shape :math:`(R, 4)`.
            The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

        """

        height = src_bbox[:, 2] - src_bbox[:, 0]
        width = src_bbox[:, 3] - src_bbox[:, 1]
        ctr_y = src_bbox[:, 0] + 0.5 * height
        ctr_x = src_bbox[:, 1] + 0.5 * width

        base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
        base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
        base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
        base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

        eps = np.finfo(height.dtype).eps
        height = np.maximum(height, eps)
        width = np.maximum(width, eps)

        dy = (base_ctr_y - ctr_y) / height
        dx = (base_ctr_x - ctr_x) / width
        dh = np.log(base_height / height)
        dw = np.log(base_width / width)

        loc = np.vstack((dy, dx, dh, dw)).transpose()
        return loc

    def generate_anchor_base(self, base_size=16, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        py = base_size / 2.
        px = base_size / 2.

        anchor_base = np.zeros((len(anchor_ratios) * len(anchor_scales), 4),
                               dtype=np.float32)
        for i in six.moves.range(len(anchor_ratios)):
            for j in six.moves.range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(anchor_ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / anchor_ratios[i])
                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.
                anchor_base[index, 1] = px - w / 2.
                anchor_base[index, 2] = py + h / 2.
                anchor_base[index, 3] = px + w / 2.
        return anchor_base

    def bbox_iou(self, bbox_a, bbox_b):
        """Calculate the Intersection of Unions (IoUs) between bounding boxes.

        IoU is calculated as a ratio of area of the intersection
        and area of the union.

        This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
        inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
        same type.
        The output is same type as the type of the inputs.

        Args:
            bbox_a (array): An array whose shape is :math:`(N, 4)`.
                :math:`N` is the number of bounding boxes.
                The dtype should be :obj:`numpy.float32`.
            bbox_b (array): An array similar to :obj:`bbox_a`,
                whose shape is :math:`(K, 4)`.
                The dtype should be :obj:`numpy.float32`.

        Returns:
            array:
            An array whose shape is :math:`(N, K)`. \
            An element at index :math:`(n, k)` contains IoUs between \
            :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
            box in :obj:`bbox_b`.

        """
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            raise IndexError

        print('bbox_a.shape:', bbox_a.shape)
        print('bbox_b.shape:', bbox_b.shape)

        # top left
        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        # bottom right
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
        area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)

faster_rcnn_boxcoder = FasterRCNNBoxCoder()
