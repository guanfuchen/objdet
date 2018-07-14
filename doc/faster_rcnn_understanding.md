# faster rcnn

---
## 结构分析

R-CNN使用神经网络解决两个主要问题：
- 在输入图像中识别可能包含前景对象的区域（感兴趣区域 - ROI）
- 计算每个ROI的对象类概率分布 - 即，计算ROI包含特定类的对象的概率。然后，用户可以选择具有最高概率的对象类作为分类结果。

R-CNN由三种主要类型的网络组成：
- Head
- Region Proposal Network (RPN)
- Classification Network

R-CNN使用预先训练的网络的前几层（例如ResNet 50）识别来自输入图像的promising特征。使用在一个数据集上训练的网络可以解决不同的问题，因为神经网络表现出“迁移学习”（Yosinski et al.2014）。网络的前几层学习检测一般特征，例如边缘和颜色斑点，这些特征是许多不同问题的良好区分特征。后面的图层学到的功能是更高级别，更具体问题的功能。可以去除这些层，或者可以在反向传播期间微调这些层的权重。从预训练网络初始化的前几层构成“头部”网络。然后，由头网络产生的卷积特征图通过区域提议网络（RPN），其使用一系列卷积和完全连接的层来产生可能包含前景对象的有希望的ROI。然后使用这些promising ROI从头网络产生的特征图中裁剪出相应的区域。这称为“Crop Pooling”。然后，通过“Crop Pooling”产生的区域通过分类网络，该分类网络学习对每个ROI中包含的对象进行分类。

### 训练

训练的目标是调整RPN和分类网络中的权重并微调头部网络的权重（这些权重从预先训练的网络（如ResNet）初始化）。回想一下，RPN网络的工作是产生最有可能的ROI和分类网络的工作是为每个ROI分配对象类分数。因此，为了训练这些网络，我们需要相应的基础事实GT，即图像中存在的对象周围的边界框的坐标和那些对象的类。这个基本事实来自免费使用的图像数据库，每个图像附带一个注释文件。此注释文件包含边界框的坐标和图像中存在的每个对象的对象类标签（对象类来自预定义对象类的列表）。这些图像数据库已被用于支持各种对象分类和检测挑战。

- Bounding Box Regression Coefficients
边界框回归系数
R-CNN的目标之一是生成紧密适合对象边界的良好边界框。 R-CNN通过采用给定的边界框（由左上角的坐标，宽度和高度定义）并通过应用一组“回归系数”来调整其左上角，宽度和高度来生成这些边界框。这些系数计算如下，其中$T_x$、$T_y$、$T_w$和$T_h$分别表示目标的top left corner的$x$，$y$坐标以及宽度和高度，另外$O_x$、$O_y$、$O_w$和$O_h$分别表示原始目标框的top left corner的$x$，$y$坐标以及宽度和高度。$t_x=\frac{(T_x-O_x)}{O_x}$，$t_y=\frac{(T_y-O_y)}{O_y}$，$t_w=\log\frac{T_w}{O_w}$，$t_h=\log\frac{T_h}{O_h}$。该函数是容易可逆的，即，给定左上角的回归系数和坐标以及原始边界框的宽度和高度，可以容易地计算目标框的左上角和宽度和高度。注意，边界框的形状没有改变 - 即，在这种变换下矩形仍然是一个矩形。
- Intersection over Union (IoU) Overlap
边界框重叠IOU
我们需要测量给定边界框与另一个边界框的接近程度，该边界框与所使用的单位（像素等）无关，以测量边界框的尺寸。该测量应该是直观的（两个重合的边界框应该具有1的重叠，并且两个非重叠的框应该具有0的重叠）并且快速且容易计算。常用的重叠度量是“联合交叉（IoU）重叠，计算如下所示”。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/iou_calc.png)

在软件实现中，R-CNN执行分为几个层，如下所示。一个层封装了一系列逻辑步骤，这些步骤可能涉及通过其中一个神经网络运行数据和其他步骤，例如比较边界框之间的重叠，执行非最大值抑制等。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rcnn_layer.png)

- Anchor Generation Layer描点生成层
该层通过首先生成9个不同比例和纵横比的锚点，然后通过在跨越输入图像的均匀间隔的网格点上平移它们来复制这些锚点，从而生成固定数量的“锚点”（边界框）
- Proposal Layer区域建议层
根据边界框回归系数变换锚点以生成变换锚点。然后通过使用锚点作为前景区域的概率应用非最大抑制来修剪锚点的数量，也就是通过$t_x=\frac{(T_x-O_x)}{O_x}$，$t_y=\frac{(T_y-O_y)}{O_y}$，$t_w=\log\frac{T_w}{O_w}$，$t_h=\log\frac{T_h}{O_h}$来计算描点（和bounding box的IOU超过一定阈值）的$t_x$，$t_y$，$t_w$和$t_h$。
- Anchor Target Layer描点目标层
锚点目标层的目标是产生一组“好”锚和相应的前景/背景标签和目标回归系数以训练区域提议网络。该层的输出仅用于训练RPN网络，并且不被分类层使用。给定一组锚点（由锚点生成层生成，锚点目标层识别有前途的前景和背景锚点。有前景的前景锚点是那些与某些地面实况框重叠高于阈值的那些。背景框是与任何重叠的那些地面实况框低于阈值。锚定目标层还输出一组边界框回归量，即每个锚目标离最近边界框的距离的度量。这些回归量只对前景框有意义，因为没有背景框的“最接近的边界框”的概念。
- RPN Loss
损失函数是一个组合：（1）RPN生成的边界框的比例被正确分类为前景/背景（2）预测回归系数与目标回归系数之间的距离度量
- Proposal Target Layer区域建议目标层
提案目标层的目标是修剪提案图层生成的锚点列表，并生成特定于类的边界框回归目标，这些目标可用于训练分类层以生成良好的类标签和回归目标
- ROI Pooling Layer ROI池化层
实现空间变换网络，该网络在给定由提议目标层产生的区域提议的边界框坐标的情况下对输入要素图进行采样。这些坐标通常不在整数边界上，因此需要基于插值的采样。
- Classification Layer分类层
分类层获取ROI池层产生的输出特征图，并将它们传递给一系列卷积层。输出通过两个完全连接的层馈送。第一层为每个区域提议生成类概率分布，第二层生成一组特定于类的边界框回归量。
- Classification Loss分类Loss
与RPN损失类似，分类损失是在优化期间最小化以训练分类网络的度量。在反向传播期间，误差梯度也流向RPN网络，因此训练分类层也修改RPN网络的权重。分类损失是以下组合：（1）RPN生成的边界框的比例被正确分类（作为正确的对象类）（2）预测回归系数与目标回归系数之间的距离度量。

---
## 参考资料

- [Notes: From Faster R-CNN to Mask R-CNN](https://www.yuthon.com/2017/04/27/Notes-From-Faster-R-CNN-to-Mask-R-CNN/) Faster RCNN和Mask RCNN的笔记总结。
- [Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) 目前来说总结最为详细的RCNN系列文章，其对应相关代码实现。
- [Understanding Faster R-CNN for Object Detection](https://ardianumam.wordpress.com/2017/12/16/understanding-faster-r-cnn-for-object-detection/) 台湾 EECS Dept of NCTU 实验室的学生对于faster RCNN系列的讲解，值得参考。
- [弄懂目标检测（Faster R-CNN）？看这篇就够了！](http://pancakeawesome.ink/%E5%BC%84%E6%87%82%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(Faster-R-CNN))
- [Faster R-CNN论文翻译——中文版](http://noahsnail.com/2018/01/03/2018-01-03-Faster%20R-CNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/) 相关论文翻译。
