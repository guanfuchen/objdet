# faster rcnn

---
## 网络模型变迁

rcnn系列网络从RCNN、fast RCNN、faster RCNN到mask RCNN，从速度、精度和功能上不断提高，在目标检测领域内的应用非常广泛，如下图所示为RCNN网络结构的示意图。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rcnn_arch_example.png)

---
## 结构分析

faster R-CNN使用神经网络解决两个主要问题：
- 在输入图像中识别可能包含前景对象的区域（感兴趣区域 - ROI）
- 计算每个ROI的对象类概率分布 - 即，计算ROI包含特定类的对象的概率。然后，用户可以选择具有最高概率的对象类作为分类结果。

faster R-CNN由三种主要类型的网络组成：
- Head
- Region Proposal Network (RPN)
- Classification Network

R-CNN使用预先训练的网络的前几层（例如ResNet 50）识别来自输入图像的promising特征。使用在一个数据集上训练的网络可以解决不同的问题，因为神经网络表现出“迁移学习”（Yosinski et al.2014）。网络的前几层学习检测一般特征，例如边缘和颜色斑点，这些特征是许多不同问题的良好区分特征。后面的图层学到的功能是更高级别，更具体问题的功能。可以去除这些层，或者可以在反向传播期间微调这些层的权重。从预训练网络初始化的前几层构成“头部”网络。然后，由头网络产生的卷积特征图通过区域提议网络（RPN），其使用一系列卷积和完全连接的层来产生可能包含前景对象的有希望的ROI。然后使用这些promising ROI从头网络产生的特征图中裁剪出相应的区域。这称为“Crop Pooling”。然后，通过“Crop Pooling”产生的区域通过分类网络，该分类网络学习对每个ROI中包含的对象进行分类。

### 训练

训练的目标是调整RPN和分类网络中的权重并微调头部网络的权重（这些权重从预先训练的网络（如ResNet）初始化）。回想一下，RPN网络的工作是产生最有可能的ROI和分类网络的工作是为每个ROI分配对象类分数。因此，为了训练这些网络，我们需要相应的基础事实GT，即图像中存在的对象周围的边界框的坐标和那些对象的类。这个基本事实来自免费使用的图像数据库，每个图像附带一个注释文件。此注释文件包含边界框的坐标和图像中存在的每个对象的对象类标签（对象类来自预定义对象类的列表）。这些图像数据库已被用于支持各种对象分类和检测挑战。

#### 先验知识
- Bounding Box Regression Coefficients
边界框回归系数
R-CNN的目标之一是生成紧密适合对象边界的良好边界框。 R-CNN通过采用给定的边界框（由左上角的坐标，宽度和高度定义）并通过应用一组“回归系数”来调整其左上角，宽度和高度来生成这些边界框。这些系数计算如下，其中$T_x$、$T_y$、$T_w$和$T_h$分别表示目标的top left corner的$x$，$y$坐标以及宽度和高度，另外$O_x$、$O_y$、$O_w$和$O_h$分别表示原始目标框的top left corner的$x$，$y$坐标以及宽度和高度。$t_x=\frac{(T_x-O_x)}{O_x}$，$t_y=\frac{(T_y-O_y)}{O_y}$，$t_w=\log\frac{T_w}{O_w}$，$t_h=\log\frac{T_h}{O_h}$。该函数是容易可逆的，即，给定左上角的回归系数和坐标以及原始边界框的宽度和高度，可以容易地计算目标框的左上角和宽度和高度。注意，边界框的形状没有改变 - 即，在这种变换下矩形仍然是一个矩形。
- Intersection over Union (IoU) Overlap
边界框重叠IOU
我们需要测量给定边界框与另一个边界框的接近程度，该边界框与所使用的单位（像素等）无关，以测量边界框的尺寸。该测量应该是直观的（两个重合的边界框应该具有1的重叠，并且两个非重叠的框应该具有0的重叠）并且快速且容易计算。常用的重叠度量是“联合交叉（IoU）重叠，计算如下所示”。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/iou_calc.png)

在软件实现中，R-CNN执行分为几个层，如下所示。一个层封装了一系列逻辑步骤，这些步骤可能涉及通过其中一个神经网络运行数据和其他步骤，例如比较边界框之间的重叠，执行非最大值抑制等。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rcnn_layer.png)

#### 网络层分析
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

#### 网络层细节

- Anchor Generation Layer描点生成层
锚生成层产生一组边界框（称为“锚框”），其具有在整个输入图像上扩展的不同大小和纵横比。这些边界框对于所有图像是相同的，即，它们不知道图像的内容。其中一些边界框将包围前景对象，而大多数不会。 RPN网络的目标是学习识别哪些框是好框 - 即，可能包含前景对象并产生目标回归系数，当应用于锚框时，将锚框转换为更好的边界框（更紧密地拟合封闭的前景对象）。
![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/boxes_anchor.png)
- Region Proposal Layer区域建议层
对象检测方法需要输入一个“区域建议系统”，它产生一组稀疏或密集的一组特征。 R-CNN系统的第一个版本使用选择性搜索方法来生成区域提议。在当前版本（称为“更快的R-CNN”）中，使用基于“滑动窗口”的技术（在前一部分中描述）来生成一组密集候选区域，然后使用神经网络驱动的区域提议网络。根据包含前景对象的区域的概率对区域提议进行排名。区域提案图层有两个目标：（1）从锚点列表中，识别背景和前景锚点（2）通过应用一组“回归系数”来修改锚点的位置，宽度和高度，以提高锚点的质量（例如，使它们更好地适应对象的边界）
区域提议层由区域提议网络和三个层组成 - 提议层，锚点目标层和提议目标层。以下各节将详细介绍这三个层。
	- ##### Region Proposal Network
	区域提议层运行头网络通过卷积层（代码中称为rpn_net），然后是RELU生成的特征映射。 rpn_net的输出通过两（1,1）个核卷积层运行，以产生背景/前景类分数和概率以及相应的边界框回归系数。头网络的步幅与生成锚点时使用的步幅相匹配，因此锚箱的数量与区域提议网络产生的信息1-1对应。
	![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rpn_arch.png)
    - ##### Proposal Layer
    提议层获取由锚生成层生成的锚框，并通过基于前景分数应用非最大抑制来修剪框的数量。它还通过将RPN生成的回归系数应用于相应的锚框来生成变换的边界框。
    - ##### Anchor Target Layer
    锚目标层的目标是选择可用于训练RPN网络的有希望的锚点：（1）区分前景和背景区域
    （2）为前景框生成良好的边界框回归系数。
    - ##### Calculating RPN Loss
    请记住，RPN层的目标是生成良好的边界框。要从一组锚框中执行此操作，RPN图层必须学会将锚框分类为背景或前景，并计算回归系数以修改前景锚框的位置，宽度和高度，使其成为“更好”的前景框（更贴近前景对象）。 RPN Loss的制定方式是鼓励网络学习这种行为。
    RPN损失是分类损失和边界框回归损失的总和。分类损失使用交叉熵损失惩罚错误分类的框，回归损失使用真实回归系数之间的距离函数（使用最接近的前景锚框匹配地面实况框计算）和网络预测的回归系数。
    $RPN_{Loss}=Classification_{Loss}+Bounding\ Box\ Regression_{Loss}$
    ###### Classification Loss
    cross_entropy(predicted _class, actual_class)
    ###### Bounding Box Regression Loss:
    $L_{loc}=\sum_{u \in all\ foreground\ anchors}l_u$
    对所有前景锚点的回归损失求和。为背景锚点执行此操作没有意义，因为背景锚点没有关联的GT Boxes。
    $l_u=\sum_{i \in x,y,w,h}smooth_{L1}(u_i(predicted)-u_i(target))$
    其中$smooth_{L1}$的计算如下，这里$\sigma$通常设置为3.
    $smooth_{L1}(x)=\frac{\sigma^2 x^2}{2}(||x||<\frac{1}{\sigma^2}) or ||x||-\frac{0.5}{\sigma}(otherwise)$
    因此，要计算损失，我们需要计算以下数量：（1）类标签（背景或前景）和锚箱的分数（2）目标回归系数为前景锚框。
    我们现在将遵循锚定目标图层的实现，以查看这些数量的计算方式。我们首先选择位于图像范围内的锚框。然后，通过首先计算所有锚箱（在图像内）与所有地面实况框的IoU（交叉联合）重叠来选择好的前景框。使用此重叠信息，两种类型的框被标记为前景：（1）类型A：对于每个地面实况框，所有具有最大IoU的前景框与地面实况框重叠（2）类型B：与某些地面实况框最大重叠的锚框超过阈值，具体如下图所示：
    ![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/gt_cls.png)
    注意，仅选择与某些地面实况框重叠的锚框超过阈值作为前景框。这样做是为了避免向RPN提供学习离最佳匹配地面实况框太远的框的回归系数的“绝望学习任务”。类似地，重叠小于负阈值的框被标记为背景框。并非所有不是前景框的框都标记为背景。既不是前景也不是背景的框标记为“不关心”。这些框不包括在RPN损失的计算中。
    - ##### Calculating Classification Layer Loss
	与RPN损失类似，分类层损失有两个组成部分 - 分类loss和边界框回归损失
    $Classification\ Layer_{Loss}=Classification_{Loss}+Bounding\ Box\ Regression_{Loss}$
    RPN层和分类层之间的主要区别在于，RPN层只处理两个类 - 前景和背景，但分类层处理我们的网络正在训练分类的所有对象类（加上背景）。分类损失是以实际对象类别和预测类别得分为参数的交叉熵损失。它的计算方法如下所示。
    - ##### Proposal Target Layer
    提案目标层的目标是从提议层输出的ROI列表中选择有希望的ROI。这些有希望的ROI将用于从头层产生的特征图执行裁剪池并传递到网络的其余部分（head_to_tail），其计算预测的类别得分和框回归系数。
    - #### ROI Pooling Layer
    ROI Pooling Layer全称为Region of Interest pooling，是先进行ROI映射然后再池化，映射是把用来训练的图像的ROI映射到最后一层特征层。具体来说，图片经过特征提取后，到最后一层卷积层时，整个feature map是原始图片的1/16，那么把ROI的4个坐标都乘以1/16转换为这个卷积层上对应的坐标，得到ROI再最后一层卷积层的坐标后，把这个ROI区域均分成HxW份，每一份进行池化，最后把池化的每一份concatenate输入到下一层。通过ROI Pooling Layer操作，所有的ROI，不论ROI大小，生成的都是固定长度的一个向量给下一层。
    ROI池化层使用最大池化将任何有效感兴趣区域内的特征转换为具有固定空间范围HxW（例如，7x7）的小特征映射，其中H和W是层超参数 这与任何特定的ROI无关。在本文中，RoI是一个转换为转换特征映射的矩形窗口。
    RoI max pooling通过将hxw RoI窗口划分为大约h / Hxw / W的子窗口的HxW网格，然后将每个子窗口中的值最大化为相应的输出 网格单元。 池中独立应用于每个要素图通道，如标准最大池中所示。 RoI层只是SPPnets使用的空间金字塔池层的特例，其中只有一个金字塔层。
    如下图所示可以看出RoI在Fast RCNN中的应用结构，其中fast-rcnn-VGG16-test.prototxt可视化可参考[fast-rcnn-VGG16-test.prototxt](https://gist.github.com/kudo1026/09c53e89ccc50c5adedc013d2b852698)
    ![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/fast_rcnn_arch.png)
    
    
#### 网络层抽象整体结构

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/faster_rcnn_abstract_arch.png)

通过以上抽象整体结构，代码实现如下所示：

```python

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
            roi_indices: 可能存在的前景roi的索引值，shape为(R，)
        """
        img_size = x.size[2:] # x.size为B*C*H*W

        # 利用CNN提取图片特征features（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
        feature = self.extractor(x) # 网络特征提取器提取特征

        # RPN区域建议网络通过图像的scale不同生成不同的anchor，以及rois，输出背景和前景scores，以及对应的前景的locs，同时对于组成输入到RPN Head的roi索引输出
        # 前向传播中RPN负责提供候选区域rois（每张图给出大概2000个候选框），也就是rois和相应的提供的2000个左右的候选框的indices
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(feature, img_size, scale) # 通过输入特征和相应的原图大小和尺度变换大小输入rois

        # head部分是将提供的2000个左右的候选框通过ROIPooling进行类分数打分和每类位置微调回归
        roi_cls_locs, roi_scores = self.head(feature, rois, roi_indices)

        # 前向传播最后输出roi类相关的locs，roi类置信分数scores，rois区域建议，roi_indices，区域建议训练indices
        # 最后输出rois和roi_indices表示候选的那些roi以及对应的微调locs和每类scores
        # 负责对rois分类和微调。对RPN找出的rois，判断它是否包含目标，并修正框的位置和坐标
        return roi_cls_locs, roi_scores, rois, roi_indices
```

#### 网络特征提取层

网络特征提取层输入图像，输出特征，主要结构改进自分类网络VGG16、ResNet等等，下图所示为基于VGG16的网络特征提取层改进思路。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/faster_rcnn_vgg16_extractor.png)

    
### 参考资料

- [roi pooling层](https://www.cnblogs.com/ymjyqsx/p/7587051.html)
- [Notes on Fast RCNN](http://shuokay.com/2016/05/18/fast-rcnn/)
- [ROI Pooling as nn layers](https://github.com/pytorch/pytorch/issues/4946) pytorch实现ROI Pooling网络层。
- [Pytorch中RoI pooling layer的几种实现](https://www.cnblogs.com/king-lps/p/9026798.html) 可参考[RoiPooling.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RoiPooling.cpp)，其中参考[pytorch examples roi_pooling.py](https://github.com/pytorch/examples/blob/d8d378c31d2766009db400ac03f41dd837a56c2a/fast_rcnn/roi_pooling.py#L38-L53)。
    
---
## caffe安装

### config.py

该文件指定了用于fast rcnn训练的默认config选项，不能随意更改，如需更改，应当用yaml再写一个config_file，然后使用cfg_from_file(filename)导入以覆盖默认config。cfg_from_file(filename)定义见该文件。
tools目录下的绝大多数文件采用--cfg 选项来指定重写的配置文件（默认采用默认的config）。See tools/{train,test}_net.py for example code that uses cfg_from_file() See experiments/cfgs/*.yml for example YAML config override files

- [ubuntu16.04配置py-faster-rcnn（CPU版）](https://blog.csdn.net/u013989576/article/details/72667245) 在mac上按照该步骤进行安装运行。
- [py-faster-rcnn代码阅读2-config.py](https://www.cnblogs.com/alanma/p/6800944.html) [py-faster-rcnn代码阅读1-train_net.py & train.py](https://www.cnblogs.com/alanma/p/6802835.html) 对照这个来参考阅读。

---
## R-CNN训练比较

R-CNN训练较Faster R-CNN慢很多，具体数值如下所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/Screen_Shot_2018-07-15_16.52.57.png)

---
## 网络可视化

网络可视化主要使用netscope，下面是不同R-CNN系列的prototxt：
- [fast_rcnn VGG16 test.prototxt](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/coco/VGG16/fast_rcnn/test.prototxt)
- [faster_rcnn VGG16 test.prototxt](https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/coco/VGG16/faster_rcnn_end2end/test.prototxt)

---
## 多任务Loss

对于每一个训练RoI，都由gt类别标签u和gt目标框回归目标v标注。然后在每一个标注的RoI上应用一个多任务loss L来联合训练分类和bbox回归：

$$L(p, u, t^u, v)=L_{cls}(p,u)+\lambda [u \ge 1]L_{loc}(t^u, v)$$

其中$p=(p_0, \cdots, p_K)$表示最后的softmax类别输出（$K+1$类），$t^k=(t_x^k, t_y^k, t_w^k, t_h^k)$表示最后的目标框微调bboxes回归偏移输出。所以多任务loss的输入包括网络的输出$t^u$（类别u的bboxes回归偏移输出）、（K+1）类别输出概率$p$、该RoI标注的真实gt类别和该RoI标注的需要微调的boxes回归输出目标v。其中包括分类计算loss（cross entropy loss）和回归计算loss（smooth L1 loss），具体$L_{cls}(p,u)=-\log p_u$是对于真实类别u的log loss。

第二个任务loss，$L_{loc}$通过类u的真实bbox回归目标$v=(v_x, v_y, v_w, v_h)$和预测的偏移目标$t^u=(t_x^u, t_y^u, t_w^u, t_h^u)$计算得到。其中该loss仅仅当非背景类别才有效，因为背景木白哦RoIs没有真实gt目标框，因此L_loc被忽略。对于bbox回归，使用smooth L1损失函数：

$$L_{loc}(t^u, v)=\sum_{i \in {x,y,w,h}}{smooth_{L_1}(t_i^u-v_i)}$$

遍历所有x,y,w,h都计算对应的smooth L1损失然后相加作为最后的bbox回归损失。

$$smooth_{L_1}(x)=0.5 x^2\ if\ |x|<1$$
$$ =|x|-0.5\ otherwise,$$

其中smooth L1损失比R-CNN和SPPnet网络中使用的L2损失对于异常值更不敏感。当回归目标不能边界时，训练L2损失通常需要精心的调整学习了来阻止梯度爆炸。而smooth L1损失则降低了敏感性。

多任务回归loss中计算的超参数$\lambda$控制了分类和回归这两个任务的平衡。本文将真实gt回归目标归一化到0均值和单位方差。其中所有的实验设置都使用$\lambda=1$。

## 小批量采样

微调期间，每一个SGD mini-batch由N=2图像构造而成，通常在整个数据集中遍历随机采样。本文使用R=128的mini-batches，每一张图像采样64个RoIs。其中目标建议框采样比例和背景建议框采样比例为1：3，25%的RoIs都是和gt目标建议框的IoU超过0.5，这些RoIs组成了前景目标类，也就是$u \ge 1$，剩下的RoIs从IoU为$[0.1, 0.5)$的建议框选取，这些RoIs组成了背景目标类，也就是$u = 0$，这里使用更低的阈值0.1作用和困难样本挖掘的作用一样，是为了针对那些目标附近的区域更难检测，而目标外的区域好检测的假设。另外训练期间概率为0.5对图像进行水平翻转操作来进行数据增广。

## RoI池化层反向传播

这里对$N=1$的mini-batch下的RoI池化网络层反向传播路由梯度进行介绍。$N>1$的情况可以直接扩展，因为前向传播中对于图像都是独立对待。

$$\frac{\partial{L}}{\partial{x_i}}=\sum_{r}\sum_{j}{[i=i^{*}(r,j)]\frac{\partial{L}}{\partial{y_{rj}}}}$$

其中$x_i$是第$i$个输入到RoI池化层的激活输入，并且$y_{rj}$来自于第r个RoI的第$j$个输出。RoI池化层计算$y_{rj}=x_{i*(r,j)}$，其中$i*(r,j)=argmax_{i^{\prime} \in R(r,j)}x_{i^{\prime}}$，其中$R(r,j)$是输入子窗口的输入集合。单个$x_i$也许会赋值到不同的输出$y_{rj}$中。

上述反向传播解释：对于每一个mini-batch RoI r和每一个池化输出$y_{rj}$，如果这个单元被最大池化操作选中那么累加为1，在反向传播中，偏导数$\frac{\partial{L}}{\partial{y_{rj}}}$通过RoI pooling网络层上层backwards函数计算得到。


![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/roi_pooling.png)

## SGD训练超参数

softmax分类网络和bbox回归全连接层分别初始化为0均值方差为0.01和0.001的高斯分布。偏置为0。全局学习率为0.001，其中权重的学习率为lr，偏置的学习率为2*lr。在VOC数据集上训练时，上述配置训练跌倒30k mini-batch迭代，然后降低学习率为0.0001并且继续训练10k迭代。当在更大的数据集上训练时，运行SGD更多迭代。同时使用了0.9的动量和0.0005的参数decay。

## 尺度不变能力

为了学习尺度不变的目标检测能力：（1）通过粗暴学习（2）通过使用图像金字塔。

粗暴学习中，每一张图像在训练和测试期间都处理为预先定义的像素大小。该网络直接从训练数据中学习尺度不变目标检测能力。

多尺度方法相反通过使用图像金字塔提供了估计的尺度不变能力。

## Fast R-CNN检测

一旦对快速R-CNN网络进行微调，检测仅比运行正向传递多一点（假设对象提议是预先计算的）。在测试阶段，R典型取值为2000，每一张图像选取R个目标建议进行打分（这些目标建议没有先验目标知识，所以如何保证2000歌目标建议基本包括了目标）。

对于每一个测试RoI r，前向传播输出了类后验概率分布p和预测到相对r的bbox偏移（K类每一个目标都有自己的精修的bbox预测）。我们使用估计得概率赋值检测置信度到r，$Pr(class=k|r)=p_k$。然后**对于每一类独立使用**非极大值抑制输出检测结果。

## Truncated SVD

Truncated SVD用来实现更快的目标检测。该方法基于观察，对于整张图像分类，fc网络层耗时较conv网络层小。但是在目标检测（Fast R-CNN），RoIs处理的数量非常大，并且几乎一半的前向传播时间在fc网络层的计算中（每一个RoIs都需要单独进行fc计算，不共享）。大的全连接网络层容易使用truncated SVD奇异值分解压缩从而得到加速。

该技术主要讲权重矩阵W分解为：

$$W \approx U \Sigma_t V^T $$

其中$U$是左奇异值向量，$V$是右奇异值向量，$\Sigma_t$是包含奇异值的对角矩阵。那么原先计算的参数$uv$降低为$t(u+v)$，大大加速了检测速度。具体是将全连接网络层分解为两个全连接网络层，第一个全连接网络层权重矩阵为$\Sigma_t V^T$，没有偏置，第二个全连接网络层权重矩阵为$U$，偏置赋值为$W$的偏置。

截断的SVD可以将检测时间减少30％以上，mAP下降很小（0.3％），并且在模型压缩后无需执行额外的微调。如果在压缩后再次进行微调，则可以进一步加速mAP的小幅下降。

## 主要结果

Fast R-CNN在VOC12数据集中获得了最高的mAP精度为65.7%，使用额外的数据可以提升到68.4%。当模型在VOC07和VOC12训练集中训练时，Fast R-CNN的mAP提升到了68.8%。

## 微调的网络层

对于SPPnet网络（较浅的网络）来说，仅仅微调全连接网络层似乎对于性能的提升很有帮助。本文基于的假设是该结果并不适用于非常深的网络。为了验证微调卷积层对于提升性能也是重要的，本文freeze 13个卷积层，仅仅学习全连接层，但是精度从66.9%下降到了61.4%，这表明通过RoI pooling网络层的学习对于深层网络精度的提升非常重要。

以上实验表明卷积层的微调也是重要的，但是并不意味着需要对于所有的卷积层进行微调。对于小网络来说（S和M），发现conv1是通用的和任务无关的。允许conv1学习对于提升mAP来说没有较大的影响。对于VGG16模型来说，发现仅仅从conv3_1到之后的9个卷积层进行微调是必须的。也是因为如下的观察，首先conv2_1的更新减缓了训练速度（1.3倍），但是精度的提升仅仅只有0.3个点，conv1_1的更新浪费了GPU内存。所以本文的实现，对于VGG16来说，非跳了conv3_1和以上的卷积层，对于S和M模型微调了conv2和以上的卷积层。

具体如**网络特征提取层**章节的网络图改进所示。

## RPN网络

RPN结构如下所示，为了生成区域建议，本文在最后一层共享的全卷积网络层输出的feature map上滑动一个小的网络。这个网络是对于输入卷积feature map的nxn的空间窗口。每一个sliding window将feature map映射到更低维度的向量中（ZF为256-d，VGG为512-d）。这个更低维度的向量输入到两个子网络中：一个box回归网络层（reg）和一个box分类网络层（cls），都是1x1的卷积层。由于在最后一层feature map上的感受野比较大因此本文使用了n=3的滑动窗口卷积层。ReLUs在nxn的卷积网络层输出应用。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rpn_arch_1.png)

## 平移不变的锚点

每一个sliding-windows位置，同时预测k个区域建议，因此reg回归网络层输出4k编码了k个boxes。cls分类网络层输出2k对每一个目标建议编码估计目标／非目标的概率。毫无疑问每一个锚点都在sliding window的中心，并且具有对应的尺度和纵横比。本文使用了3个尺度和3个纵横比，总共每一个sliding位置上有k=9个锚点。对于WxH大小的feature map，总共有WHk个锚点。我们方法的一个重要property是平移不变性，具体体现在锚点和计算相对于锚点的区域建议函数。

如果移动了一张图像中的一个物体，这proposal应该也移动了，而且相同的函数可以预测出未知的proposal。MultiBox不具备如此功能平移不变性可以减少模型大小。

## 学习区域建议的Loss函数
为了训练RPNs，对于每一个锚点都赋值了一个二分类标签（存在目标或者不存在目标）。这里赋值两种锚点为正例样本：（i）和一个gt真实box有最高的IoU，或者（ii）和gt真实box的IoU比例超过0.7（用来增加正例样本，同时这个box可以通过bbox reg精调边界框）。这里需要注意gt真实box也许赋值不同的锚点正例样本标签。将所有和gt真实boxes的IoU比例低于0.3的锚点标注为负例样本。对于非负非正的锚点不用来进行计算（相当于这些是介于object和non-object的边缘建议，对于最后的输出结果不好）。

基于以上定义，本文通过下述的多任务loss来最小化目标函数，具体定义如下：

$$L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_{i}{L_{cls}(p_i,p_i^{*})}+\lambda \frac{1}{N_{reg}}\sum_{i}{p_i^{*}L_{reg}(t_i, t_i^{*})}$$

其中i是锚点索引，$p_i$锚点i预测的目标的概率，gt真实标签$p_i^{*}$当锚点是正例样本时赋值为1，如果锚点是负例样本时赋值为0。$t_i$表示了预测的bbox坐标偏移向量，$t_i^{*}$是gt真实box的坐标偏移向量。（**这里有个问题，就是一个锚点可能对应两个或多个gt真实值吗**）。分类loss的定义$L_{cls}(p_i,p_i^{*})$是两类的log loss，回归loss使用smooth L1损失。其中$p_i^{*}L_{reg}$意味着回归loss仅仅对于正例样本锚点被激活计算，否则为0。cls和reg网络层分别对应了$\{p_i\}$和$\{t_i\}$，这两项通过$N_{cls}$和$N_{reg}$正则化，同时任务平衡loss权重为$\lambda$。

对于回归网络层，采用了如下的4个坐标参数化表示：

$$t_x=(x-x_a)/w_a$$
$$t_y=(y-y_a)/h_a$$
$$t_w=log(w/w_a)$$
$$t_h=log(h/h_a)$$
$$t_x^{*}=(x^{*}-x_a)/w_a$$
$$t_y^{*}=(y^{*}-y_a)/h_a$$
$$t_w^{*}=log(w^{*}/w_a)$$
$$t_h^{*}=log(h^{*}/h_a)$$

其中x，y，w和h表示box中心，宽度和高度。这可以被看作是从一个锚点box到临近的gt真实box的bbox回归。

## 优化

本文使用和Fast R-CNN相似的方法来采样训练RPN网络，每一个mini-batch中包含了许多的正例样本和负例样本锚点。可以对所有的锚点进行优化，但是因为负例样本占了主要成分所以将导致bias。相反，这里在一个mini-batch中的一张图像中随机采样256个锚点来计算loss函数吧，其中正例和负例样本锚点比例为1:1。如果在一张图像中有少于128个正例样本，那么用负例样本pad mini-batch。

本文使用0均值0.01标准差的高斯分布随机初始化新的网络层。所有其他网络层都通过ImageNet分类训练模型初始化。本文微调了ZF net的所有网络层，对于VGG网络层仅仅微调了conv3_1及以上（节省GPU内存）。在PACAL数据集上，对于60k的mini-batches使用0.001的学习率，接下来的20k mini-batches学习率为0.0001。优化器参数中动量为0.9，weight decay为0.0005。

## 共享目标建议和目标检测的卷积层

上述没有考虑到基于目标检测的CNN训练RPN网络。下面描述RPN和检测网络Fast R-CNN如何共享卷积特征学习。

独立训练的RPN和快速R-CNN都将以不同方式修改其卷积层。因此需要设计一个策略能够在两个网络之间共享参数而不是学习两个不同的网络。定义一个同时包含RPN和Fast R-CNN联合BP优化的单一网络并不容易。其中一个原因是Fast R-CNN训练依赖于**固定**的目标建议。

虽然这种联合优化是未来工作的一个有趣问题，但我们开发了一种实用的4步训练算法，通过交替优化来学习共享特征。

### 4-step交替优化策略：
- 在第一步中，我们如上所述训练RPN。 该网络使用ImageNet预先训练的模型进行初始化，并针对区域提议任务进行微调端对端。 
- 在第二步中，我们使用由步骤1 RPN生成的提议通过快速R-CNN训练单独的检测网络。 该检测网络也由ImageNet预训练模型初始化。 此时，两个网络不共享转换层。 
- 在第三步中，我们使用检测器网络来初始化RPN训练，但我们修复了共享转换层并仅微调RPN特有的层。 现在这两个网络共享转换层。 
- 最后，保持共享转换层固定，我们微调快速R-CNN的fc层。 因此，两个网络共享相同的转换层并形成统一的网络。

其实思路很简答，第一步训练RPN网络（feature map来自于预训练ImageNet），第二步训练Fast R-CNN（feature map来自于预训练ImageNet，RoIs来自于第一步的RPN网络），第三步微调RPN网络（feature map来自于第二步Fast R-CNN训练，仅仅微调RPN网络），第四部微调Fast R-CNN（feature map来自于第三部微调RPN网络，RoIs来自于第三步的RPN网络，仅仅微调Fast R-CNN重的fc网络层）。


### 实现细节

锚点使用的尺度为128，256和512，纵横比为1:1，1:2和2:1。

**需要小心处理跨越图像边界的anchor boxes**。 在训练期间，我们忽略所有cross-boundary锚点，因此它们不会计算到代价函数。 对于典型的1000×600图像，总共将有大约20k（60x40x9）的锚。 由于忽略了cross-boundary的锚点，每个图像大约有6k个锚点用于训练。 **如果在训练中不忽略cross-boundary异常值，则会在目标中引入大的，难以纠正的误差项，并且训练不会收敛。** 然而，在测试期间，我们仍然将完全卷积RPN应用于整个图像。这可能会生成cross-boundary建议框，我们将其剪切到图像边界。

**一些RPN建议彼此高度重叠**。 **为了减少冗余，我们根据其cls分数对提议区域采用非最大抑制（NMS）**。 我们将NMS的IoU阈值修正为0.7，这使得每个图像的建议区域大约为2k。 正如我们将要展示的那样，NMS不会损害最终的检测准确性，但会大大减少建议的数量。 在NMS之后，我们使用排名前N的建议区域进行检测。

### Faster R-CNN试验

在VOC2007+12训练集训练RPN和检测网络达到的mAP为73.2%。


---
## 面试问题

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/faster_rcnn_problem_1.png)

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/faster_rcnn_problem_2.png)

---
## 参考资料

- [Notes: From Faster R-CNN to Mask R-CNN](https://www.yuthon.com/2017/04/27/Notes-From-Faster-R-CNN-to-Mask-R-CNN/) Faster RCNN和Mask RCNN的笔记总结。
- [Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) 目前来说总结最为详细的RCNN系列文章，其对应相关代码实现，结合[chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)来看，其中官方仓库[chainercv faster_rcnn](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/faster_rcnn)。
- [Understanding Faster R-CNN for Object Detection](https://ardianumam.wordpress.com/2017/12/16/understanding-faster-r-cnn-for-object-detection/) 台湾 EECS Dept of NCTU 实验室的学生对于faster RCNN系列的讲解，值得参考。
- [弄懂目标检测（Faster R-CNN）？看这篇就够了！](http://pancakeawesome.ink/%E5%BC%84%E6%87%82%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(Faster-R-CNN))
- [Faster R-CNN论文翻译——中文版](http://noahsnail.com/2018/01/03/2018-01-03-Faster%20R-CNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/) 相关论文翻译。
- [faster-rcnn-vgg16-test.prototxt](https://gist.github.com/Immiora/1a1445677088929a2fd03f18e5f29d31) caffe netscope可视化。
- [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn)和对应的python版本[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) 简化R-CNN的Pytorch实现，复制原论文的性能。
- [从编程实现角度学习Faster R-CNN（附极简实现）](https://zhuanlan.zhihu.com/p/32404424) 中文Faster R-CNN实现博客，非常值得参考。
- [fast_rcnn pytorch官方仓库示例代码](https://github.com/pytorch/examples/tree/d8d378c31d2766009db400ac03f41dd837a56c2a/fast_rcnn)
- [FasterRCNN代码解读](http://blog.younggy.com/2018/01/24/FasterRCNN%E4%BB%A3%E7%A0%81%E8%A7%A3%E8%AF%BB/)
- [物体检测之从RCNN到Faster RCNN](https://blog.csdn.net/Young_Gy/article/details/78873836) 对RCNN、Fast RCNN和Faster RCNN网络的讲解非常详细。
- [FasterRCNN代码解读](https://blog.csdn.net/Young_Gy/article/details/79155011) 以上两篇文章都是对simple faster rcnn pytorch代码的相应解读，非常有参考价值。
- [从编程实现角度学习Faster R-CNN（附极简实现）](https://zhuanlan.zhihu.com/p/32404424)
- [caffe frcnn](https://github.com/makefile/frcnn) Faster R-CNN / R-FCN C++ version based on Caffe 基于Caffe的Faster R-CNN和R-FCN