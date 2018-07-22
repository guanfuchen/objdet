# -*- coding: utf-8 -*-
import torch
import torchvision
import skimage

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    """
    :param img: 预处理图像，输入为ndarray，C*H*W，其中将尺度缩放到最小size为600或者最大size为1000
    :param min_size: 图像预处理后最小边为600
    :param max_size: 图像预处理后最大变为1000
    :return:
    """
    C, H, W = img.shape

    # 使得满足最小边为600或者最大边为1000
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.

    # 缩放图像满足以上尺度限制，使得最小边大于600，最大边大于1000
    img = skimage.transform.resize(img, (C, H * scale, W * scale), mode='reflect')

    # 这里使用pytorch特征提取器，因此按照pytorch的方式预处理图像
    img_norm = pytorch_normalze(img)
    return img_norm