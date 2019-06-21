# 基本概念


## 逻辑回归

### 1 定义

​	假设有一个二分类问题，输出为$y∈{0,1}$，而线性回归模型产生的预测值为$z = w^Tx + b$是实数值，我们希望有一个理想的阶跃函数来帮我们实现值到0/1值的转化。然而该函数不连续，我们希望有一个单调可微的函数来供我们使用，于是便找到了Sigmoid function来替代：
$$
\phi (z) = \dfrac{1}{1 + e^{-z}}
$$
![201-ml-basics-01](201-ml-basics/201-ml-basics-01.png)

有了Sigmoid fuction之后，由于其取值在(0,1)，我们就可以将其视为类1的后验概率估计$p(y=1|x)$。如果有了一个测试点$x$，那么就可以用Sigmoid fuction算出来的结果来当做该点$x$属于类别1的概率大小。 Logistic回归主要在流行病学中应用较多，比较常用的情形是探索某疾病的危险因素，根据危险因素预测某疾病发生的概率，等等。例如，想探讨胃癌发生的危险因素，可以选择两组人群，一组是胃癌组，一组是非胃癌组，两组人群肯定有不同的体征和生活方式等。这里的因变量就是是否胃癌，即“是”或“否”，自变量就可以包括很多了，例如年龄、性别、饮食习惯、幽门螺杆菌感染等。自变量既可以是连续的，也可以是分类的。

### 2 逻辑回归的目标函数

​	如果使用误差平方和来当损失函数的话，如下公式：$J(w) = \sum_{i} \dfrac{1}{2} (\phi(z^{(i)}) - y^{(i)})^2$其中，$z^{(i)} = w^Tx^{(i)} + b$，i表示第i个样本点，y(i)表示第i个样本的真实值，$ϕ(z(i))$表示第i个样本的预测值。 这时，如果我们将$\phi (z^{(i)}) = \dfrac{1}{1 + e^{-z^{(i)}}}$代入的话，会发现这时一个非凸函数，这就意味着代价函数有着许多的局部最小值，这不利于我们的求解。

​	所以，当我们要找到逻辑回归的损失函数的时候，一般采用极大似然方法。$ϕ(z)$可以视为类1的后验估计，所以我们有
$$
p(y=1|x;w) = \phi(w^Tx + b)=\phi(z)
$$
$$
p(y=0|x;w) = 1 - \phi(z)
$$

​	其中，$p(y=1|x;w)$表示给定w，上面两式可以写成一般形式：
$$
p(y|x;w)=\phi(z)^{y}(1 - \phi(z))^{(1-y)}
$$
​	  接下来我们就要用极大似然估计来根据给定的训练集估计出参数w：
$$
L(w)=\prod_{i=1}^{n}p(y^{(i)}|x^{(i)};w)=\prod_{i=1}^{n}(\phi(z^{(i)}))^{y^{(i)}}(1-\phi(z^{(i)}))^{1-y^{(i)}}
$$
 	为了简化运算，我们对上面这个等式的两边都取一个对数 ：
$$
l(w)=lnL(w)=\sum_{i = 1}^n y^{(i)}ln(\phi(z^{(i)})) + (1 - y^{(i)})ln(1-\phi(z^{(i)}))
$$
​	在前面添加一个负号，就可以使用梯度下降法求得最小损失函数：
$$
J(w)=-l(w)
$$



### 3 求解目标函数
​	使用梯度下降法求解目标函数：
$$
\begin{align}\dfrac{\partial J(w)}{w_j} &= -\sum_{i=1}^n (y^{(i)}\dfrac{1}{\phi(z^{(i)})}-(1 - y^{(i)})\dfrac{1}{1-\phi(z^{(i)})})\dfrac{\partial \phi(z^{(i)})}{\partial w_j} \\ \quad   \quad \quad &=-\sum_{i=1}^n (y^{(i)}\dfrac{1}{\phi(z^{(i)})}-(1 - y^{(i)})\dfrac{1}{1-\phi(z^{(i)})})\phi(z^{(i)})(1-\phi(z^{(i)}))\dfrac{\partial z^{(i)}}{\partial w_j} \\ \quad \quad \quad &=-\sum_{i=1}^n (y^{(i)}(1-\phi(z^{(i)}))-(1-y^{(i)})\phi(z^{(i)}))x_j^{(i)} \\ \quad \quad \quad &=-\sum_{i=1}^n (y^{(i)}-\phi(z^{(i)}))x_j^{(i)}\end{align}
$$
​	权重更新公式：
$$
w_j := w_j + \Delta w_j,\ \Delta w_j = -\eta \dfrac{\partial J(w)}{\partial w_j}
$$
$$
w_j :=w_j+\eta \sum_{i=1}^n (y^{(i)}-\phi(z^{(i)}))x_j^{(i)}
$$