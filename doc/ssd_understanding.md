# ssd_understanding

---
## 实现问题

### 卷积层输出
这里引入了dilation，计算输出结构如下（其中floor表示向下取整，ceil表示向上取整）：

input: (N,C_in,H_in,W_in) 
output: (N,C_out,H_out,W_out)
$$H_{out}=floor((H_{in}+2padding[0]-dilation[0](kernerl\_size[0]-1)-1)/stride[0]+1)$$

$$W_{out}=floor((W_{in}+2padding[1]-dilation[1](kernerl\_size[1]-1)-1)/stride[1]+1)$$

[torch.nn.Conv2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)

### VGG16网络

参考可视化网络结构[VGG ILSVRC 16 layers](http://ethereon.github.io/netscope/#/preset/vgg-16)，绘制VGG16。

---
## 参考资料

- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab) SSD思路整理相关博客。
- [ssd_keras](https://github.com/pierluigiferrari/ssd_keras)
- [ssd_keras](https://github.com/rykov8/ssd_keras/blob/master/ssd.py) 这个SSD的keras框架实现较为清楚。
- [论文阅读：SSD: Single Shot MultiBox Detector](https://blog.csdn.net/u010167269/article/details/52563573) 相关博客。
- [torchcv SSD](https://github.com/kuangliu/torchcv/blob/master/torchcv/models/ssd/net.py) 该仓库提供了SSD300和SSD512模型，同时也是faceboxes采用的结构之一，主要参考该仓库，进行代码的**搬运**。
- [物体检测论文-SSD和FPN](http://hellodfan.com/2017/10/14/%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87-SSD%E5%92%8CFPN/) 该博客对SSD的细节介绍较好，其中主要参考对anchor(prior)的解释。
- [[Learning Note] Single Shot MultiBox Detector with Pytorch — Part 1](https://towardsdatascience.com/learning-note-single-shot-multibox-detector-with-pytorch-part-1-38185e84bd79) part1-3，解释得相对浅显，可以参考。
- [深度学习论文笔记：SSD](http://jacobkong.github.io/posts/3118967289/)
- [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) IOU计算。

---
## 网络架构图

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/ssd_arch.png)