# ssd_understanding

---
## 摘要

我们提出了一种使用单个深度神经网络检测图像中的对象的方法。我们的方法名为SSD，**将边界框的输出空间离散化为一组默认框，分布在不同的宽高比和每个要素图位置的比例上**。**在预测时，网络为每个默认框中的每个对象类别的存在生成分数，并产生对框的调整以更好地匹配目标形状**。此外，**网络将来自具有不同分辨率的多个特征图的预测组合在一起，以自然地处理各种尺寸的目标**。 SSD相对于需要目标提议的方法而言是简单的，因为它完全消除了提议生成和后续像素或特征重采样阶段，并将所有计算封装在单个网络中。这使得SSD易于训练并且可以直接集成到需要检测组件的系统中。 PASCAL VOC，COCO和ILSVRC数据集的实验结果证实，SSD与使用额外目标建议步骤的方法相比具有竞争力，并且速度更快，同时为训练和推理提供了统一的框架。对于300x300输入，SSD在VOC2007测试中获得74.3％mAP，在Nvidia Titan X上为59 FPS，对于512x512输入，SSD达到76.9％mAP，优于同类最先进的Faster R-CNN模型。与其他单级方法相比，即使输入图像尺寸较小，SSD也具有更高的精度。

## 模型

SSD方法基于前向卷积网络生成一个固定大小的bboxes和对应目标类别的scores，然后紧跟着NMS步骤来生成最后的检测。early网络层基于高质量的图像分类网络标准架构，被称为base网络。然后增加额外的结构生成检测：

### 多尺度特征图

在base网络层最后增加卷积特征网络层。**这些网络层的尺寸逐渐减小（这里可以对应分类器的输入卷积结构）**，并允许在多个尺度上预测检测。卷积模型中用于预测检测结果的每个特征层是不同的。

### 卷积预测器

每个添加的特征网络层（或可选地来自基础网络的现有特征网络层）可以**使用一组卷积滤波器生成一组固定的检测预测（这里可以对应模型架构图中的分类器）**。对于具有$p$个通道的大小为$mxn$的特征层，用于预测潜在检测参数的基本元素是3×3xp的小内核， 生成类别的分数，或相对于默认框default box坐标的形状偏移。

### 默认框default box和纵横比

本文关联了每一个特征图单元和default bounding boxes。

- 步骤1：以feature map上每个点的中点为中心（offset=0.5），生成一些同心的default box；
- 步骤2：default box最小边长为$min\_size$，最大边长为$\sqrt{min\_size*max\_size}$；
- 步骤3：增加纵横比为$aspect\_ratio$和$1/aspect\_ratio$的default box，长宽分别为$\sqrt{1/aspect\_ratio} * min\_size$和$\sqrt{aspect\_ratio} * min\_size$；
- 步骤4：每一个feature map对应的default box的min_size和max_size由公式$s_k=s_{min}+\frac{s_{max}-s_{min}}{m-1}(k-1)$计算，其中$S_{min}=0.2,S_{max}=0.9$，也就是第一层feature map对应的$min\_size=S_1, max\_size=S_2$，第二层feature map对应的$min\_size=S_2, max\_size=S_3$，以此类推。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/ssd_default_box.png)


## 训练

训练SSD和一个典型的使用区域建议的检测器的关键区别在于gt真实值信息需要被赋值到检测器输出的具体输出中。训练同时包括选择default boxes和scales集合，同时也包括negative mining和数据augmentation策略。

### 匹配策略

在训练期间，我们需要确定哪些default boxes对应于gt真实检测并相应地训练网络。 对于每个gt真实目标框，我们从default boxes中进行选择，这些框的位置，宽高比和比例都有所不同。 我们首先将每个gt真实目标框与具有最佳jaccard重叠的default boxes匹配。 与MultiBox不同，我们将default boxes与jaccard重叠高于阈值（0.5）的任何gt真实目标框匹配。 这简化了学习问题，允许网络预测多个重叠default boxes的high scores，而不是要求它仅选择具有最大重叠的框。

### 训练目标函数

令$x_{ij}^{p}={1,0}$表示对于目标p第i个default box和第j个真实gt目标框的匹配指示器。整体的训练目标损失函数是定位localization损失和置信度confidence损失的加权和。其中损失函数计算包括$x$（和default box的匹配，1位正例样本匹配，0为负例样本匹配），$c$是输出的预测框（d）目标类别结果，$l$是输出的预测框（d）的偏移量，$g$是gt真实值的目标框。

$$L(x, c, l, g)=\frac{1}{N}((L_{conf}(x,c)+\alpha L_{loc}(x,l,g)))$$

其中N是匹配的default boxes数量。如果N=0，设置loss为0。localization loss是预测box（l）和真实目标框（g）的Smooth L1损失。和Faster R-CNN相似，本文回归了相对default boxes（d）的中心、宽度和高度的偏移。

如下计算，当$x_{ij}^{k}=1$，也就是匹配时，才计算预测box以及真实目标框的smooth L1定位损失。

#### 定位损失

$$L_{loc}(x, l, g)=\sum_{i \in Pos}^{N} \sum_{{m \in {cx, cy, w, h}}}{x_{ij}^k smooth_{L1}(l_i^m-\hat{g}_j^m)}$$

$$\hat{g}_j^{cx}=(g_j^{cx}-d_i^{cx})/d_i^{w}$$

$$\hat{g}_j^{cy}=(g_j^{cy}-d_i^{cy})/d_i^{h}$$

$$\hat{g}_j^{w}=\log(\frac{g_j^{w}}{d_i^{w}})$$

$$\hat{g}_j^{h}=\log(\frac{g_j^{h}}{d_i^{h}})$$

#### 置信度损失

置信度confidence损失是多类置信度下的softmax loss，其中对于i是正例样本的default box，计算正例样本的default box，如果是负例样本的default box，计算负例样本的default box，既不是正例样本也不是负例样本则忽略。

$$L_{conf}(x,c)=-\sum_{i \in Pos}^{N}{x_{ij}^{p} \log{(\hat{c}_{i}^{p})}}-\sum_{i \in Neg}^{N}{ \log{(\hat{c}_{i}^{0})}}$$

$$\hat{c}_{i}^{p}=\frac{\exp{c_{i}^{p}}}{\sum_{p}{\exp{c_{i}^{p}}}}$$

### 选择default boxes的尺度和纵横比

为了处理不同的目标尺度，一些方法建议以不同的尺寸处理图像并在之后组合结果。 但是，通过利用单个网络中几个不同层的特征图进行预测，我们可以模拟相同的效果，同时还可以**跨所有目标尺度共享参数**。在这些方法的推动下（FCN skip connections），我们使用lower和upper的特征图进行检测。在一个网络内来自于不同levels的特征图有不同大小的感受野。本文假设使用m个特征图进行预测，每一个特征图的default boxes尺度计算参考**默认框default box和纵横比**章节。最后每一个特征图feature map位置上共有6个default boxes，包括纵横比（1，2，3，1/2，1/3），同时对于纵横比为1的情况下增加了一个新的尺度$s_k^{\prime}=\sqrt{{s_k}{s_{k+1}}}$

### 困难样本挖掘
Hard negative mining，匹配步骤以后，大部分的default boxes都是负例样本。这就引入了正例样本和负例样本的不平衡。通过使用最大的置信度损失（较难分类）的default box并且选择整理了样本和负例样本，使得比例为1:3，这个策略使得更好的优化以及更稳定的训练。

### 数据增广

为了使得对不同目标大小shapes的适应，每一个训练图像按照下述选项随机采样：
- 使用整张原始输入图像；
- 采样和原目标有0.1，0.3，0.5，0.7或0.9的patch；
- 随机采样patch；

每一个采样patch的大小是原始输入图像大小的$[0.1, 1]$，并且纵横比在1/2和2之间。如果gt真实框的中心在采样patch的中心那么保持gt真实目标框。采样步骤后，每一个采样patch被resized到固定大小并且0.5概率的水平翻转，同时额外增加一些photo-metric畸变。

## 实验结果

本文的实验都是基于VGG16，改进如下所示：
- 转换fc6和fc7为卷积网络层；
- 池化pool5从2x2-s2转换为3x3-s1，同时使用空洞卷积来增强丢失的感受野；
- 去除所有dropout网络层和fc8网络层；

SSD300模型中，使用了conv4_3,conv7,conv8_2,conv9_2,conv10_2,conv11_2这6个不同stride的feture map来预测location和confidences。其中在conv4_3中设置default box尺度为0.1。使用xavier方法初始化参数。对于conv4_3,conv10_2和conv11_2，仅仅关联4个default boxes（舍弃aspect ratios 1/3和3）。**由于conv4_3和其他网络层相比有不同的特征尺度，本文使用了L2 norm缩放特征norm到20并且在BP期间学习scale**。在40k的迭代过程中使用学习率0.0001，然后继续使用0.0001和0.00001的学习率微调10k迭代。SSD300模型比Fast R-CNN精度更高，SSD512模型比Faster R-CNN精度更高（在VOC 2007提升1.7%mAP）。

SSD对于bbox大小非常敏感，在更小的目标上比起更大的目标性能更差，因为那些非常小的目标甚至在非常top的网络层中也没有足够的信息。减小这种更差性能的问题，可以增加输入大小（从300x300增加到512x512）能帮助提升检测小的目标。好的一面是，SSD对于大的目标表现较好，并且对于不同的目标纵横比表现较为鲁棒（因为在每一个feature map位置上使用了不同纵横比的default boxes）。

### 数据增广

Fast和Faster R-CNN使用原始图像和水平翻转来训练。 我们使用更广泛的采样策略，类似于YOLO。我们不知道我们的采样策略对Fast和Faster R-CNN有多大益处，但它们可能会受益更少，因为它们在分类期间使用特征池化步骤，这对于设计的目标平移相对稳健。

### 更多的default box形状

使用不同的aspect ratio，比如2,1/2,3,1/3都会提升精度。

### 空洞卷积

使用了空洞卷积和直接池化层下采样的比较：精度相同但是速度提升了20%。

### 不同分辨率的多个输出网络层

SSD的主要贡献是在不同的输出层上使用不同比例的默认框。

## 运行时间

考虑到从我们的方法生成的大量目标框，必须在测试期间有效地执行非最大抑制（nms）。通过使用0.01的置信度阈值，我们可以过滤掉大多数bbox。然后，我们应用nms，每个类的IOU重叠为0.45，并保持每个图像的前200个检测。 对于SSD300和20个VOC，此步骤每张图像的成本约为1.7毫秒，这接近于在所有新添加的层上花费的总时间（2.4毫秒）。

我们的SSD300和SSD512方法在速度和精度方面均优于Faster R-CNN。尽管Fast YOLO 的运行速度为155FPS，但精度较低，几乎可达到22％mAP。据我们所知，SSD300是第一个实现70％以上mAP的实时方法。请注意，大约80％的转发时间花在基础网络上（在我们的例子中是VGG16）。因此，使用更快的基础网络甚至可以进一步提高速度，这也可能使SSD512模型达到实时运行。


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

- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)，网络结构可以参考[caffe ssd train.prototxt](https://github.com/intel/caffe/blob/master/models/intel_optimized_models/ssd/VGGNet/coco/SSD_300x300/train.prototxt)
- [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab) SSD思路整理相关博客。
- [ssd_keras](https://github.com/pierluigiferrari/ssd_keras)
- [ssd_keras](https://github.com/rykov8/ssd_keras/blob/master/ssd.py) 这个SSD的keras框架实现较为清楚。
- [论文阅读：SSD: Single Shot MultiBox Detector](https://blog.csdn.net/u010167269/article/details/52563573) 相关博客。
- [torchcv SSD](https://github.com/kuangliu/torchcv/blob/master/torchcv/models/ssd/net.py) 该仓库提供了SSD300和SSD512模型，同时也是faceboxes采用的结构之一，主要参考该仓库，进行代码的**搬运**。
- [物体检测论文-SSD和FPN](http://hellodfan.com/2017/10/14/%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87-SSD%E5%92%8CFPN/) 该博客对SSD的细节介绍较好，其中主要参考对anchor(prior)的解释。
- [[Learning Note] Single Shot MultiBox Detector with Pytorch — Part 1](https://towardsdatascience.com/learning-note-single-shot-multibox-detector-with-pytorch-part-1-38185e84bd79) part1-3，解释得相对浅显，可以参考。
- [深度学习论文笔记：SSD](http://jacobkong.github.io/posts/3118967289/)
- [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) IOU计算。
- [ssds.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch) 该仓库实现了大量的SSD变种（pytorch），**参考**。
- [SSD详解Default box的解读](https://blog.csdn.net/wfei101/article/details/78597442)

---
## 网络架构图

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/ssd_arch.png)