# objdet

---
## object detection algorithms

这个仓库旨在实现常用的目标检测算法，主要参考如下：
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [Object-Detection](http://songit.cn/Object-Detection.html)
- [2015-10-09-object-detection.md](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-object-detection.md) handong收集的相关目标检测论文目录；
- [awesome-object-detection](https://github.com/amusi/awesome-object-detection)，awesome系列，参考合并了handong相关目录；
- ...

---
### 论文资料

- [DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling](https://arxiv.org/abs/1703.10295) 相关代码[denet](https://github.com/lachlants/denet)。
- [Soft Proposal Networks for Weakly Supervised Object Localization](https://arxiv.org/pdf/1709.01829.pdf) 相关代码[SPN.pytorch](https://github.com/yeezhu/SPN.pytorch)
- [ICCV 2015 Tutorial on Tools for Efficient Object Detection](http://mp7.watson.ibm.com/ICCV2015/ObjectDetectionICCV2015.html) ICCV 2015中举办的关于目标检测的教程，可以参考。
- [Deep Learning for Objects and Scenes](http://deeplearning.csail.mit.edu/) CVPR 2017关于目标检测的教程。
- [RSA-for-object-detection-cpp-version](https://github.com/QiangXie/RSA-for-object-detection-cpp-version) [RSA-for-object-detection](https://github.com/sciencefans/RSA-for-object-detection) 相关论文[Recurrent Scale Approximation for Object Detection in CNN](https://arxiv.org/pdf/1707.09531.pdf)
- DetNet: A Backbone network for Object Detection
- 小目标检测，参考如下

---
### 小目标检测

- [Feature-Fused SSD: Fast Detection for Small Objects](https://arxiv.org/abs/1709.05054)
- [Perceptual Generative Adversarial Networks for Small Object Detection](https://arxiv.org/abs/1706.05274)，GAN小目标检测，暂且不看；
- [Detecting and counting tiny faces](https://arxiv.org/abs/1801.06504)
- [Seeing Small Faces from Robust Anchor's Perspective](https://arxiv.org/abs/1802.09058)
- [Face-MagNet: Magnifying Feature Maps to Detect Small Faces](https://arxiv.org/abs/1803.05258)
- [Small-scale Pedestrian Detection Based on Somatic Topology Localization and Temporal Feature Aggregation](https://arxiv.org/abs/1807.01438)
- [MDSSD: Multi-scale Deconvolutional Single Shot Detector for Small Objects](https://arxiv.org/abs/1805.07009)
- [CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection](https://arxiv.org/abs/1606.05413) 集成人体上下文信息来帮助推理人脸位置；
- [Finding Tiny Faces](https://arxiv.org/abs/1612.04402)，多级图像金字塔进行multi-scale训练和测试；
- [S3FD: Single Shot Scale-invariant Face Detector](https://arxiv.org/abs/1708.05237) single shot尺度等变的人脸检测器；

---
### 网络实现

- SSD，[ssd_understanding](doc/ssd_understanding.md)
- Faster RCNN，[faster_rcnn_understanding](doc/faster_rcnn_understanding.md)
- R-FCN，[rfcn_understanding]()
- ...

---
### 非极大值抑制

- [soft-nms](https://github.com/bharatsingh430/soft-nms)

---
### 困难样例学习

- Training Region-based Object Detectors with Online Hard Example Mining，非常有效的针对Regin-based目标检测模型的在线困难样例学习策略。

---
### 数据集实现

- COCO
- ...

---
### 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)

```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```

---
### TODO

- 实现数据集加载VOC

