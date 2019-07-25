# 图像信号的数学表示

​	数字图像处理就是把真实世界中的连续三维随机信号投影到传感器的二维平面上，采样并量化后得到二维矩阵；通过对二维矩阵的处理，从二维矩阵图像中恢复出三维场景，这正是计算机视觉的主要任务之一。于是可以将图像处理分两个阶段：第一阶段：信号处理阶段也就是采样、量化的过程，图像处理其实就是二维和三维信号处理，而处理的信号又有一定的随机性，因此经典信号处理和随机信号处理都是图像处理和计算机视觉中必备的理论基础。第二阶段就是数学处理阶段，图像处理涉及到了微积分 、矩阵、概率论等相关数学知识。

​         图像的频率是表征图像中灰度变化剧烈程度的指标，是灰度在平面空间上的梯度。如：大面积的沙漠在图像中是一片灰度变化缓慢的区域，对应的频率值很低；而对于地表属性变换剧烈的边缘区域在图像中是一片灰度变化剧烈的区域，对应的频率值较高。傅立叶变换在实际中有非常明显的物理意义，设f是一个能量有限的模拟信号，则其傅立叶变换就表示f的谱。从纯粹的数学意义上看，傅立叶变换是将一个函数转换为一系列周期函数来处理的。从物理效果看，傅立叶变换是将图像从空间域转换到频率域，其逆变换是将图像从频率域转换到空间域。换句话说，傅立叶变换的物理意义是将图像的灰度分布函数变换为图像的频率分布函数，傅立叶逆变换是将图像的频率分布函数变换为灰度分布函数。