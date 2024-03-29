# 基本概念



## 线性回归

### 1 定义

* 给定训练数据 $D=\{X_i,Y_i\}_{i=1}^N$，其中 $y\in R$，回归学习一个从输入$x$ 到输出y的映射$ f$
* 对新的测试数据$x$，用学习到的映射对其进行预测： $\hat{y}=f(x) $
* 若假设映射$ f$ 是一个线性函数，即 $y =  f  (X|\theta )= \theta^T X $
* 我们称之为线性回归模型。
对于只有一个自变量$x$和一个因变量$y$的数据集，我们假设$(x,y)$之间的关系可以用
$$
h(x)=\theta x + b
$$
进行映射，那么我们就把求解这个函数的过程称为线性回归求解。可以在二维坐标系拟合出一条直线来代表$(x,y)$的趋势。很显然，不可能所有的点刚好在这条直线上，所以求解最优的标准就是，使点集$(x,y)$与拟合函数间的误差（平方误差）最小,推广到多个自变量，假设一个数据集有$n$个自变量，也可以说成$n$个属性，如果使用线性模型去拟合则有
$$
h_{\theta}(\textbf{x}_i)= \theta_1 x_1 +\theta_2 x_2 +...+\theta_n x_n+b
$$
这种方式称为多元线性回归（multivariate linear regression）.需要求解的未知量有$θ_1,θ_2,...,θ_n,b​$。为了便于讨论，令$b=\theta_0x_0​$，其中$x_0=1​$，那么，原来的函数就可以表示为
$$
h_{\theta}(\textbf{x}_i)=\theta_0 x_0 + \theta_1 x_1 +\theta_2 x_2 +...+\theta_n x_n
$$

### 2 线性回归的目标函数

​	目标函数通常包含两项：损失函数和正则项。
$$
J(\theta)= \sum_{i=1}^N{L(y_i,f(x_i))+\lambda R(\theta)}
$$
​	求解损失函数一般用到最小二乘法,我们有很多的给定点，这时候我们需要找出一条线去拟合它，那么我先假设这个线的方程，然后把数据点代入假设的方程得到观测值，求使得实际值与观测值相减的平方和最小的参数。对变量求偏导联立便可求：
$$
J(\theta)=\sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$
​	对回归问题，损失函数可以采用L2损失，得到l岭回归模型损失函数：
$$
J_R(\theta)=\frac 12 \left \| y-X\theta\right\|^2 +\frac 12 \lambda \left\| \theta \right\|^2
$$
​	也可以使用L1损失，得到Lasso模型损失函数：
$$
J_L(\theta)=\frac 12 \left \| y-X\theta\right\|^2 +\lambda \sum \left| \theta_i \right|
$$

### 3 求解目标函数

​	现在我们的目的就是求解出一个使得代价函数最小的$θ$，梯度下降算法是一种求局部最优解的方法，对于$F(x)$，在$a$点的梯度是$F(x)$增长最快的方向，那么它的相反方向则是该点下降最快的方向。原理：将函数比作一座山，我们站在某个山坡上，往四周看，从哪个方向向下走一小步，能够下降的最快，迭代公式如下；
$$
\mathbf\theta= \mathbf\theta - \alpha\mathbf{X}^T(\mathbf{X\theta} - \mathbf{Y})
$$
​	确定好θ，得到最终的线性回归公式，只需要把测试的x输入进去就可以得到预测的y值了。
