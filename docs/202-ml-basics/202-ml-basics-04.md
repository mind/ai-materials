#  GBDT

### 提升方法AdaBoost

​	Adaboost是Boosting中较为代表的算法，基本思想是通过训练数据的分布构造一个分类器，然后通过误差率求出这个若弱分类器的权重，通过更新训练数据的分布，迭代进行，直到达到迭代次数或者损失函数小于某一阈值。 算法流程如下：

\(sdfsdf)\

1. 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),(x_N,y_N)}$，其中，$x_i∈X⊆R_n$，$y_i∈Y=−1,1$，迭代次数$M  $

2. 初始化训练样本的权值分布：
   $$
   D_1=(w_{1,1},w_{1,2},…,w_{1,i}),w_{1,i}=\frac{1}{N},i=1,2,…,N
   $$

3. 对于 $ m=1,2,…,M $

   (a)使用具有权值分布$Dm$的训练数据集进行学习，得到弱分类器$G_m(x)  $

   (b)计算$G_m(x)$在训练数据集上的分类误差率：
   $$
   e_m=\sum_{i=1}^Nw_{m,i}  I(G_m (x_i )≠y_i )
   $$
   (c)计算$Gm(x)$在强分类器中所占的权重：
   $$
   α_m=\frac{1}{2}log \frac{1-e_m}{e_m}
   $$
   (d)更新训练数据集的权值分布（这里，$z_m$是归一化因子，为了使样本的概率分布和为1）：
   $$
   w_{m+1,i}=\frac{w_{m,i}}{z_m}exp⁡(-α_m y_i G_m (x_i ))，i=1,2,…,10\\
   z_m=\sum_{i=1}^Nw_{m,i}exp⁡(-α_m y_i G_m (x_i ))
   $$

4. 得到最终分类器：
   $$
   F(x)=sign(\sum_{i=1}^Nα_m G_m (x))
   $$



### 提升树（Boosting Tree）

​	提升树是以决策树为基分类器的提升方法，通常使用CART树。针对不同问题的提升树学习算法，主要区别在于使用的损失函数不同。分类问题一般使用指数损失函数。可以使用CART分类树作为AdaBoost的基分类器，此时为分类提升树。回归问题则使用平方误差损失函数。

​	提升树模型可以表示为决策树的加法模型：
$$
f_M(x)=\sum_{m=1}^MT(x;\Theta)
$$
​	采用平方损失的回归树算法流程如下：

1. 初始化$f_0(x)=0$；

2. 对$m=1，2，3...M$ ：

   (a)计算残差$r_mi=y_i−f_{m−1}(x_i),i=1,2...N$

   (b)拟合残差$r_{mi}$学习一个回归树，得到$T(x;Θ)$

   (c)更新$f_m(x)=f_{m−1}(x)+T(x;Θ)$

3. 得到回归问题决策树。



​



### 梯度提升决策树(Gradient Boosting Decision Tree，GBDT）

#### GBDT概述

​	在上面简单介绍了Gradient Boost框架，梯度提升决策树Gradient Boosting Decision Tree是Gradient Boost框架下使用较多的一种模型，在梯度提升决策树中，其基学习器是分类回归树CART，使用的是CART树中的回归树。 在GBDT的迭代中，假设我们前一轮迭代得到的强学习器是$ft−1(x)$, 损失函数是$L(y,ft−1(x))$, 我们本轮迭代的目标是找到一个CART回归树模型的弱学习器$ht(x)$，让本轮的损失$L(y,ft(x)=L(y,ft−1(x)+ht(x))$最小。也就是说，本轮迭代找到决策树，要让样本的损失尽量变得更小。

　　GBDT的思想可以用一个通俗的例子解释，假如有个人30岁，我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。如果我们的迭代轮数还没有完，可以继续迭代下面，每一轮迭代，拟合的岁数误差都会减小。

　　从上面的例子看这个思想还是蛮简单的，但是有个问题是这个损失的拟合不好度量，损失函数各种各样，怎么找到一种通用的拟合方法呢？

#### GBDT回归算法

​	梯度提升（gradient boosting）是一种组合算法，它的基分类器是决策树，既可以用来回归，也可以用作分类。 算法其核心就在于，每棵树是从先前所有树的残差中来学习。利用的是当前模型中损失函数的负梯度值:
$$
r_{mi} = - \Bigg [ \frac {\partial L(y_i, f (x_i))}{\partial f (x_i)}\Bigg ] _{f (x) = f _{m-1}(x)}
$$
​	作为提升树算法中的残差的近似值，进而拟合一棵回归树。

​	提升树算法步骤：

​	输入：训练数据集$T=(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$,损失函数$L(y,f(x))$;

​	输出：回归树$\tilde f(x)$

1. 首先初始化：
   $$
   f_0(x)=arg min_c \sum_{i=1}^{N}L(y_i, c)
   $$
   估计一个使损失函数极小化的常数值，此时它只有一个节点的树；

2. 迭代的建立$M$棵提升树，对$ m=1,2,...,M$:

   (a) for i=1 to N： 计算损失函数的负梯度在当前模型的值，并将它作为残差的估计值：
   $$
   r_{mi} = - \Bigg [ \frac {\partial L(y_i, f (x_i))}{\partial f (x_i)}\Bigg ] _{f (x) = f _{m-1}(x)}
   $$
   (b) 对于$r_mi$拟合一棵回归树，得到第m棵树的叶节点区域$R_mj,j=1,2,…,J $

   (c) for j=1 to J 计算：
   $$
   c_{mj} = arg min_c \sum_{x_i\epsilon R_{mj}}L(y_i,f_{m-1}(x_i)+c)
   $$
   利用线性搜索估计叶节点区域的值，使损失函数极小化；

   (d) 更新:
   $$
   f_{m}(x) = f_{m-1}(x) + \sum_{j=1}^Jc_{mj}I(x \epsilon R_{mj})
   $$

3. 得到回归树：
   $$
   \tilde{f}(x)=f_M(x)=\sum_{m=1}^M\sum_{j=1}^Jc_{mj}I(x \epsilon R_{mj})
   $$






#### GBDT分类算法

1. 二元分类：

   对于二元GBDT，如果用类似于逻辑回归的对数似然损失函数，则损失函数为：
   $$
   L(y, f(x)) = log(1+ exp(-yf(x)))
   $$
   其中$y \in\{-1, +1\}$。则此时的负梯度误差为：

   $$
   r_{mi} = -\bigg[\frac{\partial L(y, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{m-1}\;\; (x)} = y_i/(1+exp(yf(x_i)))
   $$

   对于生成的决策树，我们各个叶子节点的最佳残差拟合值为：
   $$
   c_{mj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} log(1+exp(y_i(f_{m-1}(x_i) +c)))
   $$
   由于上式比较难优化，我们一般使用近似值代替：
   $$
   c_{mj} = \sum\limits_{x_i \in R_{tj}}r_{mi}\bigg /  \sum\limits_{x_i \in R_{tj}}|r_{mi}|(2-|r_{mi}|)
   $$
   除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，二元GBDT分类和GBDT回归算法过程相同。

2. 多元分类：

   多元GBDT要比二元GBDT复杂一些，对应的是多元逻辑回归和二元逻辑回归的复杂度差别。假设类别数为K，则此时我们的对数似然损失函数为：
   $$
   L(y, f(x)) = -  \sum\limits_{k=1}^{K}y_klog\;p_k(x)
   $$
   其中如果样本输出类别为k，则$y_k=1​$。第k类的概率$p_k(x) ​$的表达式为：

$$
p_k(x) = exp(f_k(x)) \bigg / \sum\limits_{l=1}^{K} exp(f_l(x))
$$

​	集合上两式，我们可以计算出第$t$轮的第$i$个样本对应类别$l$的负梯度误差为：
$$
r_{til} = -\bigg[\frac{\partial L(y, f(x_i)))}{\partial f(x_i)}\bigg]_{f_k(x) = f_{l, t-1}\;\; (x)} = y_{il} - p_{l, t-1}(x_i)
$$
​	观察上式可以看出，其实这里的误差就是样本$i$对应类别$l$的真实概率和$t-1$轮预测概率的差值。

　　对于生成的决策树，我们各个叶子节点的最佳残差拟合值为：

$$
c_{tjl} = {\underbrace{arg \; min}}_{c_{jl}}
\sum\limits_{i=0}^{m}\sum\limits_{k=1}^{K} L(y_k, f_{t-1, l}(x) + \sum\limits_{j=0}^{J}c_{jl} I(x_i \in R_{tj})
$$


​	由于上式比较难优化，我们一般使用近似值代替：

$$
c_{tjl} =  \frac{K-1}{K} \; \frac{\sum\limits_{x_i \in R_{tjl}}r_{til}}{\sum\limits_{x_i \in R_{til}}|r_{til}|(1-|r_{til}|)}
$$

​	除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，多元GBDT分类和二元GBDT分类以及GBDT回归算法过程相同。
