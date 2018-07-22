# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import six

from objdet.dataloader import faster_rcnn_data_utils
from objdet.utils.config import faster_rcnn_config
from objdet.layers.region_proposal_network import RegionProposalNetwork
from objdet.nms.py_cpu_nms import py_cpu_nms


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

faster_rcnn_boxcoder = FasterRCNNBoxCoder()

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        """
        :param extractor: FasterRCNN中特征提取网络，比如VGG16，ResNet50等
        :param rpn: FasterRCNN中区域建议网络，生成ROI
        :param head: FasterRCNN中定位分类模块，其中包括ROIPooling等
        """
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # FasterRCNN中loc loss中归一化mean和std
        self.loc_normalize_mean = [0., 0., 0., 0.]
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        self.nms_thresh = 0.3 # 非极大值抑制超参数nms，该参数在训练期间和测试期间可能不同
        self.score_thresh = 0.7 # 非极大值抑制超参数score，该参数在训练期间和测试期间可能不同

        self.num_classses = self.head.num_classses # 目标检测总数，包括背景区域，其中VOC为20+1共计21类



    def suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = py_cpu_nms(np.array(cls_bbox_l), self.nms_thresh, prob_l)
            # TODO
            # 实现non_maximum_suppression
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def forward(self, x, scale=1.0):
        """
        FasterRCNN前向传播，其中x表示输入图像变量，输出为roi_cls_loc，roi_scores，rois和roi_indices
        :param x: input image Variable
        :param scale: input image scale
        :returns:
            其中R表示输入到Head中的ROI数目，相当于batch_size，C表示总类别数
            roi_cls_locs: roi每一类的locs，shape为(R,(C*4))
            roi_scores: roi每一类的置信分数scores，shape为(R,C)
            rois: 可能存在的前景roi，shape为(R，4)
            roi_indices: 可能存在的前景roi的索引值，也就是0指背景，1指目标1，2指目标2等等，shape为(R，)
        """
        img_size = x.size[2:] # x.size为B*C*H*W

        # 利用CNN提取图片特征features（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
        feature = self.extractor(x) # 网络特征提取器提取特征

        # RPN区域建议网络通过图像的scale不同生成不同的anchor，以及rois，输出背景和前景scores，以及对应的前景的locs，同时对于组成输入到RPN Head的roi索引输出
        # 前向传播中RPN负责提供候选区域rois（每张图给出大概2000个候选框），也就是rois和相应的提供的2000个左右的候选框的indices，也就是目标类别
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(feature, img_size, scale) # 通过输入特征和相应的原图大小和尺度变换大小输入rois

        # head部分是将提供的2000个左右的候选框通过ROIPooling进行类分数打分和每类位置微调回归
        roi_cls_locs, roi_scores = self.head(feature, rois, roi_indices)

        # 前向传播最后输出roi类相关的locs，roi类置信分数scores，rois区域建议，roi_indices，区域建议训练indices
        # 最后输出rois和roi_indices表示候选的那些roi以及对应的微调locs和每类scores
        # 负责对rois分类和微调。对RPN找出的rois，判断它是否包含目标，并修正框的位置和坐标
        return roi_cls_locs, roi_scores, rois, roi_indices

    def predict(self, imgs, sizes=None):
        """
        检测imgs中的目标bounding boxes
        :param imgs: imgs的shape为B*C*H*W，图像格式为RGB格式，像素值大小为[0, 255]
        :param sizes:
        :return:
        """
        # 预测前，防止BN层等带来的问题，首先需要设置eval测试模式
        self.eval()

        prepared_imgs = list()
        sizes = list()
        for img in imgs:
            size = img.shape[1:]
            img = faster_rcnn_data_utils.preprocess(img)

            # 将预处理过的图像数据和相应的size加入到prepared_imgs和sizes列表中
            prepared_imgs.append(img)
            sizes.append(size)
        # 预测imgs的bboxes、labels和scores，也就是对应每一个图像的bbox、label和score
        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img_tensor = torch.from_numpy(img)
            img = Variable(img_tensor.float()[None], volatile=True)
            # 由于训练数据预处理，所以尺度将会进行相应的缩放，这里通过计算当前的H/原先存储的size的H来计算缩放的尺度
            scale = img.shape[3] / size[1]
            roi_cls_locs, roi_scores, rois, roi_indices = self(img, scale)

            # 这里假设了batch_size为1
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = rois.data / scale

            # 转换预测的值为对应的bounding boxes值，同时按照缩放比例复原到愿图像
            mean = torch.Tensor(self.loc_normalize_mean).repeat(self.num_classses)[None]
            std = torch.Tensor(self.loc_normalize_std).repeat(self.num_classses)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.num_classses, 4)

            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

            # 利用预测的微调loc对相应的roi进行微调输出bbox
            cls_bbox = faster_rcnn_boxcoder.loc2bbox(roi.numpy().reshape((-1, 4)), roi_cls_loc.numpy().reshape((-1, 4)))
            cls_bbox = cls_bbox.view(-1, self.num_classses * 4)

            # clip bounding box防止bounding box越界，其中y方向限制为[0, size[0]也就是h]，x方向限制为[0，size[1]也就是w]
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            # 对于最后的输出score通过softmax函数表示为概率
            prob = F.softmax(roi_score, dim=1)

            raw_cls_bbox = cls_bbox
            raw_prob = prob

            # 非极大值抑制获取最后的输出
            bbox, label, score = self.suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        # 预测完毕后继续训练模式
        self.train()
        return bboxes, labels, scores




def decomVGG16():
    model = torchvision.models.vgg16(pretrained=True)

    # 去除最后的MaxPool
    features = list(model.features)[:-1]
    classifier = list(model.classifier)
    # 其中classifier[2]和[5]为dropout，另外[6]为最后一层全连接层
    del classifier[2]
    del classifier[5]
    del classifier[6]
    # 删除后重新序列化分类器
    classifier = nn.Sequential(*classifier)


    # 固定前4层卷积
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    # 固定前4层卷积不参与反向传播并进行序列化特征器
    features = nn.Sequential(*features)

    return features, classifier

class FasterRCNNVGG16(FasterRCNN):
    """
    FasterRCNNVGG16是以VGG16为特征提取器的FasterRCNN
    """
    feat_stride = 16 # VGG16卷积5的输出为原先分辨率的1／16

    def __init__(self, num_classes=21, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        """
        构造FasterRCNNVGG16目标检测模型
        :param num_classes: 不包括背景在内的目标类别
        :param anchor_ratios: 生成锚点的ratios比例
        :param anchor_scales: 生成锚点的scales尺度
        """
        # 使用VGG16网络构建的网络特征提取器和分类器
        extractor, classifier = decomVGG16()

        # 区域建议网络
        rpn = RegionProposalNetwork(512, 512, anchor_ratios=anchor_ratios, anchor_scales=anchor_scales, feat_stride=self.feat_stride)

        # 类相关区域分类微调网络
        head = VGG16ROIHead(num_classes=num_classes, roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=classifier)

        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)


# # 实现RPN网络
# class RegionProposalNetwork(nn.Module):
#     def __init__(self, in_channels=512, mid_channels=512, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16):
#         """
#         区域建议网络，通过特征输入生成类无关的区域建议，也就是背景和目标
#         :param in_channels: 输入到RPN的特征通道数
#         :param mid_channels: 第一层扩大的卷积通道数
#         :param anchor_ratios: 锚点比例
#         :param anchor_scales: 锚点尺寸
#         :param feat_stride: 特征间隔
#         """
#         super(RegionProposalNetwork, self).__init__()
#         self.in_channels = in_channels
#         self.mid_channels = mid_channels
#         self.anchor_ratios = anchor_ratios
#         self.anchor_scales = anchor_scales
#         self.feat_stride = feat_stride
#
#         self.anchor_base = faster_rcnn_boxcoder.generate_anchor_base(anchor_scales=anchor_scales, anchor_ratios=anchor_ratios)
#
#     def forward(self, x):
#         pass


class RoIPooling2D(nn.Module):
    def __init__(self, out_h=7, out_w=7, spatial_scale=1.0):
        """
        roi pooling参考[roi_pooling.py](https://github.com/pytorch/examples/blob/d8d378c31d2766009db400ac03f41dd837a56c2a/fast_rcnn/roi_pooling.py#L38-L53)
        :param size:
        :param spatial_scale:
        :return:
        """
        super(RoIPooling2D, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.spatial_scale = spatial_scale

    def forward(self, x, rois):
        """
        :param x: x为feature map
        :param rois: rois表示roi的目标，shape为(R，5)，其中（cls, x, y, w, h）
        :return:
        """
        assert (rois.dim() == 2)
        assert (rois.size(1) == 5)
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)

        # 将空间尺度恢复到原图像中大小中
        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()
        adaptive_max_pool = nn.AdaptiveMaxPool2d((self.out_h, self.out_w))
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = x.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
            output.append(adaptive_max_pool(im))

        return torch.cat(output, 0)


class VGG16ROIHead(nn.Module):
    def __init__(self, num_classes, roi_size, spatial_scale, classifier):
        """
        基于VGG16的ROI Head网络，将ROI通过ROI Pooling层转换为相同roi_size的Feature Map（一般和VGG16输入fc6和fc7前的卷积维度相同，设置为7）
        :param num_classes: 包括背景的目标类，用来输出最后的roi_cls_locs和roi_scores
        :param roi_size: 通过ROI Pooling层转换后的roi_size*roi_size大小的卷积层
        :param spatial_scale: 该ROI相对于原始图像变换的空间缩放尺度大小
        :param classifier: fc6和fc7全连接分类器
        """
        super(VGG16ROIHead, self).__init__()

        # 最后输出的检测目标（包括背景）的数量
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self.roi = RoIPooling2D(out_h=self.roi_size, out_w=self.roi_size, spatial_scale=self.spatial_scale)

        # ROI输入至ROI Pooling层之后固定大小的卷积层输入的分类网络层
        self.classifier = classifier


        # 每一类的位置微调网络和置信度评分网络
        self.cls_loc = nn.Linear(4096, self.num_classes*4)
        self.score = nn.Linear(4096, self.num_classes*1)


        # ROIHead用于分类和回归的卷积层用0均值高斯初始化，同时方差分别为0.01和0.001
        self.cls_loc.weight.data.normal_(0, 0.001)
        self.cls_loc.bias.data.zero_()

        self.score.weight.data.normal_(0, 0.01)
        self.score.bias.data.zero_()

    def forward(self, x, rois, roi_indices):
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]

        pool = self.roi(x, indices_and_rois)

        # 输入全连接前将尺度resize到(B, -1)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)

        # roi的每一类的locs微调系数和置信度scores
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
