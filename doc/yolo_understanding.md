# yolo_understanding

---
## 论文
- You Only Look Once: Unified, Real-Time Object Detection，YOLO系列V1
- YOLO9000: Better, Faster, Stronger，YOLO系列V2
- YOLOv3: An Incremental Improvement，YOLO系列V3

---
## YOLO系列V1

### 摘要

我们提出了一种新的目标检测方法YOLO。先前关于物体检测的工作**重新利用分类器来执行检测**。相反，我们**将目标检测作为一个回归问题，具体是回归为空间分离的边界框和相关的类概率**。单个神经网络在一次评估中直接从完整图像预测边界框和类概率。由于整个检测流水线是单个网络，因此可以直接在检测性能上进行端到端优化。
我们的统一架构非常快。**我们的基础YOLO模型以每秒45帧的速度实时处理图像**。**较小版本的网络Fast YOLO每秒处理惊人的155帧，同时mAP仍然是其他实时检测器的的两倍**。与最先进的检测系统相比，YOLO产生更多的定位误差（localization loss），但不太可能错误地将背景预测为目标。最后，YOLO学习了目标通用的表示。当从自然图像到艺术品等其他领域进行生成时，它优于其他检测方法，包括DPM和R-CNN。

### 统一检测

我们的系统**将输入图像分成SxS网格。 如果目标的中心落入网格单元格中，则该网格单元格负责检测该目标**。**每个网格单元预测B个目标边界框以及对应的置信度分数**。这些置信度分数反应了模型多大概率确定box包含了目标已经预测的准确性。具体来说，**本文定义了confidence为$Pr(Object)*IOU_{pred}^{truth}$，如果网络中没有目标存在，那么$Pr(Object)=0$，同时置信度分数也为0。否则$Pr(Object)=1$，置信度分数为预测框和gt真实目标框的IOU**。

每一个bbox包含了5个预测值：x，y，w，h和confidence，**坐标（x,y）表示box相对于grid cell边界的中心，宽度和高度是相对于整个图像的预测值**。最后置信度阈值表示预测box和任何一个真实目标框的IOU。

每一个网格单元也预测C个条件概率密度，$Pr(Class_i|Object)$，也就是存在目标时类i的概率。这些概率是基于包含目标的网格单元的条件概率。我们仅仅对于每一个网格单元预测一个类概率集合。

测试时，我们将条件概率和单独的box置信度预测相乘作为每一个box的类相关的置信度分数，这些分数同时编码了box中类出现的概率和预测框拟合目标的好坏：
$$Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}=Pr(Class_i)*IOU_{pred}^{truth}$$

具体在VOC中，使用S=7，B=2，C=20，最后的预测是7x7x30。

下图可以直观看出YOLO的设计思路，每一个网格仅仅预测一个目标，但是可以预测多个目标框位置：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_model_1.png)

下面黄色网格预测person类目标，因为这个目标框的中心在该网络内。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_grid.jpeg)

下面黄色网格预测person类目标，同时预测多个目标框位置，这里预测两个bbox来预测人的位置。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_grid_2.jpeg)

单目标假设的规则严格限制了目标靠近的程度，下图是YOLO检测人的示意图，图中9个圣诞老人仅仅检测了5个。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_grid_3.jpeg)

对于每一个网格单元：
- 预测B个目标边界框，并且每一个box都有box置信度（IOU）；
- 无视目标框的数量检测一个目标；
- 预测C个条件类概率；

整体前向传播检测结构如下：
![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_grid_4.jpeg)

#### 网络设计

YOLO使用CNN结构，网络的初始卷积层从图像中提取特征，同时全连接层输出概率和坐标。

我们的网络架构受到用于图像分类的GoogLeNet模型的启发。我们的网络有**24个卷积层**，后面是**2个全连接层**。与GoogLeNet使用的inception模块不同，**我们只使用1x1的卷积层，然后使用3x3个卷积层，类似于Lin等**。完整的网络如图3所示。

本文同时部署了一个Fast YOLO用来快速目标检测，仅仅使用了9个卷积层和更少的滤波器。

YOLO和Fast YOLO最后的输出是7x7x30的张量预测值。

详细的网络结构设计：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_v1_arch.png)

#### 训练

我们在ImageNet 1000级竞赛数据集上预先训练我们的卷积层。**对于预训练，我们使用架构图中的前20个卷积层，接着是平均池化层和全连接层**。 我们训练这个网络大约一周，并在ImageNet 2012验证集上达到了88％的top-5精度，与Caffe's Model Zoo中的GoogLeNet模型相当。

**研究表明将卷积和连接层添加到预训练网络可以提高性能**。按照他们的例子，我们**添加了四个卷积层和两个全连接层，随机初始化权重**。检测通常需要细粒度的视觉信息，因此我们将网络的**输入分辨率从224x224增加到448x448**。

我们的**最后一层预测了类概率和边界框坐标**。 我们**将边界框宽度和高度标准化为图像宽度和高度，使它们落在0和1之间**。我们**将边界框x和y坐标参数化为特定网格单元位置的偏移量，因此它们也在0和1之间**。

**最后一层网络层使用了线性激活函数，其他网络层使用了LReLU激活函数**。

$$\phi(x)=x, if\ x>0$$
$$=0.1x, otherwise$$

最后优化了均方和误差训练模型。为了防止定位误差和分类误差不一致导致的发散问题，其中增加了bbox坐标预测loss并降低了不包含目标的分类loss。其中使用了两个参数，$\lambda_{coord}$和$\lambda_{noobj}$来实现，本文设置$\lambda_{coord}=5$，$\lambda_{noobj}=0.5$。

均方和误差同时也将大的目标框和小的目标框等价考虑。我们的误差度量应该反映出大box中的小偏差小于小box。这里通过**预测bbox的宽度和高度的均方根**来部分解决这个问题。

YOLO每个网格单元预测多个边界框。在训练时，我们只希望一个边界框预测器负责每一个目标。 我们根据哪个预测具有和当前gt真实目标框有最高IOU，指定为“负责”以预测目标。 这导致边界框预测变量之间的特化。每个预测变量都能更好地预测某些大小，宽高比或对象类别，从而提高整体召回率。

综上训练期间下述多部分的loss函数：

$$\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}[(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2]$$
$$+\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}[(\sqrt{w_i}-\sqrt{\hat{w}_i})^2+(\sqrt{h_i}-\sqrt{\hat{h}_i})^2]$$
$$+\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}(C_i-\hat{C}_i)^2$$
$$+\lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B}1_{ij}^{obj}(C_i-\hat{C}_i)^2$$
$$+\sum_{i=0}^{S^2}1_{i}^{obj}\sum_{c \in classes}(p_i(c)-\hat{p}_i(c))^2$$

其中$1_{i}^{obj}$表示目标在单元格中是否出现，$1_{ij}^{obj}$表示第j个bbox预测器在单元格是否对那个预测负责。

为了避免过拟合使用了dropout和扩展的数据增广，包括引入了随机缩放和平移，以及随机调整饱和度。

具体分析如下所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_v1_loss_analysis.png)

#### YOLO的限制

YOLO引入了bbox预测较强的空间限制，因为每一个单元格仅仅只能预测两个boxes并且只有一类。这个限制限制了模型能够预测的邻近目标数量，因此对于成群出现的小目标来说预测效果较差，比如鸟群。另外就是均方和误差loss带来的定位精度较低，因为对于大的目标框和小的目标框其weight是一样的。


---
## YOLO系列V2

### 摘要

我们介绍YOLO9000，这是一种state-of-the-art的实时目标检测系统，**可以检测超过9000个目标类别**。首先，我们提出了**对YOLO检测方法的各种改进，既有新颖的，也有先前的工作**。**改进型号YOLOv2是标准检测任务（如PASCAL VOC和COCO）的state-of-the-art**。使用新颖的**多尺度训练**方法，相同的YOLOv2模型可以以不同的尺寸运行，在速度和准确度之间提供简单的权衡。**在67FPS时，YOLOv2在VOC 2007上获得76.8 mAP。在40 FPS时，YOLOv2获得78.6 mAP，优于最先进的方法，如使用SSD和ResNet Faster R-CNN，同时仍然运行得更快**。最后，我们提出了一种**联合训练目标检测和分类的方法**。使用此方法，我们在COCO检测数据集和ImageNet分类数据集上同时训练YOLO9000。我们的联合训练允许YOLO9000预测没有标记检测数据的对象类的检测。我们验证了ImageNet检测任务的方法。YOLO9000在ImageNet检测验证集上获得19.7 mAP，尽管只有200个类中的44个具有检测数据。在不在COCO的156个班级中，YOLO9000获得16.0 mAP。但是YOLO可以检测到超过200个类别;它预测了超过9000种不同对象类别的检测。它仍然可以实时运行。

### 更好

YOLO相对于state-of-the-art的检测系统存在各种缺点。与Fast R-CNN相比，YOLO的误差分析表明，YOLO会产生**大量的定位误差**。此外，与基于区域建议的方法相比，YOLO具有**相对较低的召回率**。 因此，我们主要关注改善召回率和定位性能，同时保持分类准确性。

相较于使用更大更深的网络，YOLOV2不是扩展网络，而是**简化网络，然后使表示更容易学习**，构建更快更好的网络。

#### Batch Normalization

**使用Batch Normalization**能够加速收敛速度，同时可以去除其他正则化方法，比如dropout，本文在所有卷积网络中增加BN提升了2%mAP。

#### 高分辨率分类器

YOLOV1通过在ImageNet数据集上预训练分类模型（分辨率为224x224），然后在检测任务上微调（分辨率为448x448）。**这里增加了分类模型高分辨率微调（分辨率为448x448）**，提升了4%mAP。

#### Anchor Boxes卷积

YOLO直接使用全连接网络层预测了bounding boxes的坐标。预测偏移而不是坐标简化了问题，使网络更容易学习。

删除了全连接网络层使用anchor boxes来预测bounding boxes。首先去除了一个池化层来增加网络卷积层的输出分辨率。**减小网络使得处理416输入图像而不是448（因为在特征图中希望在中心出有一个奇数个位置）**。416像素的输入图像通过32倍下采样输出特征图为13x13。**这里使用anchor boxes尽管降低了mAP，但是提升了区域建议的召回率，意味着网络有更大的改进空间提升mAP。**

#### Dimension Clusters

这里关注使用k-means来选取anchor boxes。对于不同方法来生成prior的性能比较，可见使用聚类的方法获取priors相比手动选取的priors具有更好的结果。

#### 直接位置预测

本文预测了相对单元格的相对定位坐标。该方法加上Dimension聚类大约提升了5%mAP。

具体做法，网络在输出的feature map上预测5个bboxes，对于每一个box预测了5个坐标$t_x,t_y,t_w,t_w,t_o$：

$$b_x=\sigma(t_x)+c_x$$
$$b_y=\sigma(t_y)+c_y$$
$$b_w=p_w e^{t_w}$$
$$b_h=p_h e^{t_h}$$
$$Pr(object)*IOU(b,object)=\sigma(t_o)$$

具体示意图如下所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_v2_anchor_box.png)

#### Fine-Grained Features

这个修改过的YOLO可以预测13x13特征图上的检测结果。虽然这对于大型目标来说足够了，但它可能会受益于更细粒度的特征来定位较小的目标。Faster R-CNN和SSD都在网络中的各种feature map上运行其建议网络，以获得一系列分辨率的检测结果。**我们采用不同的方法，只需添加一个passthrough层，从较早从26x26的分辨率的网络层中获取特征**。passthrough层将更高的分辨率特征通过连接到不同不同的通道上来提升特征数。比如从特征feature map为26x26x512修改为13x13x2048，然后和原始特征连接。这使得1%mAP提升。

#### 多尺度训练

原始的YOLO使用了输入分辨率448x448。由于使用了anchor box这里修改了分辨率为416x416。由于网络是全卷积可以针对任意大小的输入突袭那个，这里为了获得尺度变化鲁棒的检测性能，使用不同大小的图像训练模型。

**具体做法，每隔10batches，网络随机选取新的图像维度大小，由于图像降采样为32，选取32倍数的图像大小：${320，352，...，608}$，因此最小的选择是320x320，最大的选择是608x608。**


YOLO和其他模型速度精度比较图：
![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_v2_speed_accuracy.png)

### 更快

修改网络结构获得更快的检测速度。

大多数检测框架依赖于VGG-16作为基本特征提取器。VGG-16是一个功能强大，准确的分类网络，但它过于复杂。VGG-16的卷积层需要306.9亿浮点运算才能在224×224分辨率的单个图像上进行单次通过。

YOLO框架使用了基于Googlenet架构修改的网络，该网络比VGG-16更快，但是性能更差，具体在ImageNet上top-5精度对比为88%:90%。

本文通过使用3x3卷积和1x1卷积以及BN等方法修改得到更好的网络模型Darknet-19，该模型有19个卷积层和5个最大池化层，详细结构如下图所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/yolo_v2_darknet19.png)

### 更强大

这部分没有细看，主要是提出了一种联合训练分类和检测数据的机制。


---
## YOLO系列V3

### 摘要

我们向YOLO提供一些更新！我们进行了一些小的设计更改，以使其更好。我们还训练了这个新网络。它比上次有点大，但更准确。它仍然很快，不用担心。在320x320，YOLOv3以22.2毫秒的速度精度为28.2mAP运行，与SSD一样准确，但速度提高了三倍。当我们查看旧的.5 IOU mAP检测指标YOLOv3非常好。它在Titan X上在51毫秒内达到57.9 $AP_{50}$，相比于RetinaNet在198毫秒内达到57.5 $AP_{50}$，性能相似，但速度提高了3.8。

### Bounding Box Prediction

和YOLO9000相同，使用维度聚类的方法选取锚点目标框来预测边界框。

$$b_x=\sigma(t_x)+c_x$$
$$b_y=\sigma(t_y)+c_y$$
$$b_w=p_w e^{t_w}$$
$$b_h=p_h e^{t_h}$$

训练期间使用平方和误差。

### Class Prediction

使用多标签分类预测每一个bbox可能包含的类别的概率。**我们不使用softmax，因为我们发现它对于提升良好的性能不是必须的，相反使用独立的逻辑分类器。在训练中，我们使用二元交叉熵损失进行类预测。**

这种策略能够应对那些有标签重叠的数据集，比如Woman和Person，使用softmax会引入类标签不重合的问题，这里使用独立的逻辑回归代替。

### Predications Across Scales跨尺度预测

和SSD类似在feature map上增加多个不同的尺度。

### Feature Extractor特征提取器

这里使用新的网络来完成特征提取，具体是Daranet-19以及shortcut connections的结合，最后的网络层共有53个卷积层，这里称为Darknet-53。

### 训练

**YOLO V3仍然没有使用困难样本挖掘训练**，其使用了**多尺度训练**，大量的**数据增广**，**BN**，以及其他标准操作。

随着IOU限制增加YOLO V3的性能大幅度下降，这也显示了YOLO V3仍然对获取较好的目标框较难。

对于检测中IOU的评价标注严格后，YOLO V3的性能下降，这表明了其对预测精确的目标框仍然较低性能。

---
## 参考资料

- [YOLO 论文阅读](https://xmfbit.github.io/2017/02/04/yolo-paper/) 这篇博客对YOLO V1-3系列的论文细节创新很好的列举出来了，非常值得参考。
- YOLO v1模型参考[yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)
- [Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3](https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088) medium针对YOLO的解读，基本涵盖了论文所有的要点。
- [目标检测（九）--YOLO v1,v2,v3](https://blog.csdn.net/App_12062011/article/details/77554288)
- [caffe-yolov3](https://github.com/ChenYingpeng/caffe-yolov3)，使用caffe实现的yolov3；