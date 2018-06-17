# objdet

---
## object detection algorithms

这个仓库旨在实现常用的目标检测算法，主要参考如下：
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- ...

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

