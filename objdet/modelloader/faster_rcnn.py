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

from torchnet.meter import ConfusionMeter, AverageValueMeter

from objdet.dataloader import faster_rcnn_data_utils
from objdet.utils.config import faster_rcnn_config
from objdet.layers.region_proposal_network import RegionProposalNetwork, AnchorTargetCreator, ProposalTargetCreator
from objdet.nms.py_cpu_nms import py_cpu_nms
from objdet.utils.faster_rcnn_boxcoder import faster_rcnn_boxcoder


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


    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = faster_rcnn_config.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': faster_rcnn_config.weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


def decomVGG16():
    model = torchvision.models.vgg16(pretrained=True)

    # 去除最后的MaxPool
    features = list(model.features)[:-1]
    classifier = list(model.classifier)
    # 其中classifier[2]和[5]为dropout，另外[6]为最后一层全连接层
    # 从后往前删除不会影响要删除的分类器模块的次序
    del classifier[6]
    del classifier[5]
    del classifier[2]
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

    def __init__(self, num_classses=21, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
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

        head = VGG16ROIHead(num_classses=num_classses, roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=classifier)

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
    def __init__(self, num_classses, roi_size, spatial_scale, classifier):
        """
        基于VGG16的ROI Head网络，将ROI通过ROI Pooling层转换为相同roi_size的Feature Map（一般和VGG16输入fc6和fc7前的卷积维度相同，设置为7）
        :param num_classses: 包括背景的目标类，用来输出最后的roi_cls_locs和roi_scores
        :param roi_size: 通过ROI Pooling层转换后的roi_size*roi_size大小的卷积层
        :param spatial_scale: 该ROI相对于原始图像变换的空间缩放尺度大小
        :param classifier: fc6和fc7全连接分类器
        """
        super(VGG16ROIHead, self).__init__()

        # 最后输出的检测目标（包括背景）的数量
        self.num_classses = num_classses
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self.roi = RoIPooling2D(out_h=self.roi_size, out_w=self.roi_size, spatial_scale=self.spatial_scale)

        # ROI输入至ROI Pooling层之后固定大小的卷积层输入的分类网络层
        self.classifier = classifier


        # 每一类的位置微调网络和置信度评分网络
        self.cls_loc = nn.Linear(4096, self.num_classses*4)
        self.score = nn.Linear(4096, self.num_classses*1)


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


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = faster_rcnn_config.rpn_sigma
        self.roi_sigma = faster_rcnn_config.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_mean, self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
        gt_rpn_label = Variable(gt_rpn_label).long()
        gt_rpn_loc = Variable(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        self.rpn_cm.add(_rpn_score, _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), gt_roi_label.long()]
        gt_roi_label = Variable(gt_roi_label).long()
        gt_roi_loc = Variable(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

        self.roi_cm.add(roi_score, gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = faster_rcnn_config.state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            faster_rcnn_config.parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: v for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum()  # ignore gt_label==-1 for rpn_loss
    return loc_loss