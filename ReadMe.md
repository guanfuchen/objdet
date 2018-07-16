# objdet

---
## object detection algorithms

这个仓库旨在实现常用的目标检测算法，主要参考如下：
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- ...

---
### 论文资料

- [DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling](https://arxiv.org/abs/1703.10295) 相关代码[denet](https://github.com/lachlants/denet)。
- [Soft Proposal Networks for Weakly Supervised Object Localization](https://arxiv.org/pdf/1709.01829.pdf) 相关代码[SPN.pytorch](https://github.com/yeezhu/SPN.pytorch)
- [ICCV 2015 Tutorial on Tools for Efficient Object Detection](http://mp7.watson.ibm.com/ICCV2015/ObjectDetectionICCV2015.html) ICCV 2015中举办的关于目标检测的教程，可以参考。
- [Deep Learning for Objects and Scenes](http://deeplearning.csail.mit.edu/) CVPR 2017关于目标检测的教程。

---
### 网络实现

- SSD，[ssd_understanding](doc/ssd_understanding.md)
- ...

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

