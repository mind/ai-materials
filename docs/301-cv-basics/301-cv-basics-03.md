# 视觉系统

计算机视觉系统的结构形式很大程度上依赖于其具体应用方向。有些是独立工作的，用于解决具体的测量或检测问题；也有些作为某个大型复杂系统的组成部分出现，比如和机械控制系统，数据库系统，人机接口设备协同工作。计算机视觉系统的具体实现方法同时也由其功能决定——是预先固定的抑或是在运行过程中自动学习调整。尽管如此，有些功能却几乎是每个计算机系统都需要具备的：

## 图像获取

一幅数字图像是由一个或多个图像感知器产生，这里的感知器可以是各种光敏摄像机，包括遥感设备，X射线断层摄影仪，雷达，超声波接收器等。取决于不同的感知器，产生的图片可以是普通的二维图像，三维图组或者一个图像序列。图片的像素值往往对应于光在一个或多个光谱段上的强度（灰度图或彩色图），但也可以是相关的各种物理数据，如声波，电磁波或核磁共振的深度，吸收度或反射度。

## 预处理

在对图像实施具体的计算机视觉方法来提取某种特定的信息前，一种或一些预处理往往被采用来使图像满足后继方法的要求。例如：

二次取样保证图像坐标的正确；

平滑去噪来滤除感知器引入的设备噪声；

提高对比度来保证实现相关信息可以被检测到；

调整尺度空间使图像结构适合局部应用。

## 特征提取

从图像中提取各种复杂度的特征。例如：

线，边缘提取；

局部化的特征点检测如边角检测，斑点检测；

更复杂的特征可能与图像中的纹理形状或运动有关。

## 检测分割

在图像处理过程中，有时会需要对图像进行分割来提取有价值的用于后继处理的部分，例如

筛选特征点；

分割一或多幅图片中含有特定目标的部分。

## 高级处理

到了这一步，数据往往具有很小的数量，例如图像中经先前处理被认为含有目标物体的部分。这时的处理包括：

验证得到的数据是否符合前提要求；

估测特定系数，比如目标的姿态，体积；

对目标进行分类。

高级处理有理解图像内容的含义，是计算机视觉中的高阶处理，主要是在图像分割的基础上再经行对分割出的图像块进行理解，例如进行识别等操作。
