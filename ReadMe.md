# objdet

---
## object detection algorithms

这个仓库旨在实现常用的目标检测算法，主要参考如下：
- [mmdetection](https://github.com/open-mmlab/mmdetection)，参考CUHK检测框架的思路。
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [Object-Detection](http://songit.cn/Object-Detection.html)
- [2015-10-09-object-detection.md](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-object-detection.md) handong收集的相关目标检测论文目录；
- [awesome-object-detection](https://github.com/amusi/awesome-object-detection)，awesome系列，参考合并了handong相关目录；
- [目标检测 Object Detection](http://www.xzhewei.com/Paper-Archives-%E8%AE%BA%E6%96%87%E9%9B%86/Object-Detection/#Is-Faster-R-CNN-Doing-Well-for-Pedestrian-Detection) 博客整理收集的相关资料；
- [Review of Deep Learning Algorithms for Object Detection](https://medium.com/comet-app/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852) 相关目标检测DL算法综述；
- [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)，其中增加了可视化paper表格，非常直观；
- Deep Learning for Generic Object Detection: A Survey，深度学习通用目标检测调研；
- ...

> 图片来自于[deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/deep_learning_object_detection_history.png)

---
### 论文资料

- [DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling](https://arxiv.org/abs/1703.10295) 相关代码[denet](https://github.com/lachlants/denet)。
- [Soft Proposal Networks for Weakly Supervised Object Localization](https://arxiv.org/pdf/1709.01829.pdf) 相关代码[SPN.pytorch](https://github.com/yeezhu/SPN.pytorch)
- [ICCV 2015 Tutorial on Tools for Efficient Object Detection](http://mp7.watson.ibm.com/ICCV2015/ObjectDetectionICCV2015.html) ICCV 2015中举办的关于目标检测的教程，可以参考。
- [Deep Learning for Objects and Scenes](http://deeplearning.csail.mit.edu/) CVPR 2017关于目标检测的教程。
- [RSA-for-object-detection-cpp-version](https://github.com/QiangXie/RSA-for-object-detection-cpp-version) [RSA-for-object-detection](https://github.com/sciencefans/RSA-for-object-detection) 相关论文[Recurrent Scale Approximation for Object Detection in CNN](https://arxiv.org/pdf/1707.09531.pdf)
- DetNet: A Backbone network for Object Detection
- 小目标检测，参考如下
- 遮挡目标检测，参考如下
- 视频目标检测，参考[video_obj](https://github.com/guanfuchen/video_obj)
- domain目标检测，参考如下
- 非极大值处理，参考如下
- 弱监督目标检测，参考如下
- 困难样本采样策略，参考如下
- 文本检测，参考如下
- 类别不平衡目标检测，参考如下
- 小数据集目标检测，参考如下
- A unified multi-scale deep convolutional neural network for fast object detection
- How Far are We from Solving Pedestrian Detection? **行人检测**
- Taking a Deeper Look at Pedestrians **行人检测**
- Integralchannel features **行人检测**
- Fast Feature Pyramids for Object Detection **行人检测**
- What Can Help Pedestrian Detection? **行人检测**
- Citypersons: A di- verse dataset for pedestrian detection **行人检测**数据集
- DenseBox: Unifying Landmark Localization with End to End Object Detection，不使用anchor的检测方法
- UnitBox: An Advanced Object Detection Network
- Discriminative models for multi-class object layout
- Learning Transferable Architectures for Scalable Image Recognition，自学网络结构；
- Single-Shot Refinement Neural Network for Object Detection，S3FD和RefineDet论文都是同一个作者；
- HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection
- Hypercolumns for Object Segmentation and Fine-grained Localization


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
### 遮挡目标检测

- Repulsion Loss: Detecting Pedestrians in a Crowd

---
### domain目标检测
- [Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation](https://naoto0804.github.io/cross_domain_detection/)

---
### 文本检测
- Deep Direct Regression for Multi-Oriented Scene Text Detection
- TextBoxes: A Fast Text Detector with a Single Deep Neural Network
- Detecting Text in Natural Image with Connectionist Text Proposal Network
- R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection
- EAST: An Efficient and Accurate Scene Text Detector
- Detecting Oriented Text in Natural Images by Linking Segments
- Arbitrary-Oriented Scene Text Detection via Rotation Proposals
- Scene Text Detection via Holistic, Multi-Channel Prediction
- Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection

---
### 类别不平衡目标检测

- Solution for Large-Scale Hierarchical Object Detection Datasets with Incomplete Annotation and Data Imbalance

---
### 小数据集目标检测

- Comparison Detector: A novel object detection method for small dataset

---
### 弱监督目标检测
- Weakly Supervised Deep Detection Networks

---
### 非极大值抑制

- Learning non-maximum suppression
- Improving Object Detection With One Line of Code，soft-nms

---
### 困难样本采样策略

- Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors

---
### 回归框loss

目前常用的回归框loss有l2 loss，smooth l1 loss和IoU loss。

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

