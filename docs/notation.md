# 附录-符号约定



为保证本书内容连贯一致，特针对内容及符号约定如下。

- 公式
  - 行内和行间公式均采用双\$\$作为起止符号。
- 图片
  - 所有模块单独建立一个md文件，其内所有章节放在以模块md文件同名的目录中。
  - 所有章节均单独新建一个md文件，文件名以课程大纲中知识点编号为文件名。
  - 每个md附带的图片，放到md同名文件夹中，并以md文件名开始编号，如201-mk-skills.md对应图片文件夹为201-mk-skills，其中图片文件名为201-mk-skills-01.jpg，201-mk-skills-02.png...


# 数学符号
本节简要介绍本书所使用的数学符号。
我们在至中描述大多数数学概念，如果你不熟悉任何相应的数学概念，可以参考对应的章节。


##  数和数组
|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle a$  |           标量 (整数或实数)|
|$\displaystyle \boldsymbol{\mathit{a}}$  |向量|
|$\displaystyle \boldsymbol{\mathit{A}}$  |          矩阵|
|$\displaystyle {\textbf{A}}$  |          张量|
|$\displaystyle \boldsymbol{\mathit{I}}_n$  |         $n$行$n$列的单位矩阵|
|$\displaystyle \boldsymbol{\mathit{I}}$  |          维度蕴含于上下文的单位矩阵|
|$\displaystyle \boldsymbol{\mathit{e}}^{(i)}$  |       标准基向量$[0,\dots,0,1,0,\dots,0]$，其中索引$i$处值为1|
|$\displaystyle \text{diag}(\boldsymbol{\mathit{a}})$  |    对角方阵，其中对角元素由$\boldsymbol{\mathit{a}}$给定|
|$\displaystyle \mathrm{a}$  |          标量随机变量|
|$\displaystyle \mathbf{a}$  |          向量随机变量|
|$\displaystyle \boldsymbol{\mathrm{A}}$  |          矩阵随机变量|


##  集合和图


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle \mathbb{A}$  |         集合|
|$\displaystyle \mathbb{R}$  |         实数集|
|$\displaystyle \{0, 1\}$  |       包含0和1的集合|
|$\displaystyle \{0, 1, \dots, n \}$  |  包含$0$和$n$之间所有整数的集合|
|$\displaystyle [a, b]$  |        包含$a$和$b$的实数区间|
|$\displaystyle (a, b]$  |        不包含$a$但包含$b$的实数区间|
|$\displaystyle \mathbb{A} \backslash \mathbb{B}$  |差集，即其元素包含于$\mathbb{A}$但不包含于$\mathbb{B}$|
|$\displaystyle \mathcal{G}$  |         图|
|$\displaystyle Pa_\mathcal{G}(\mathrm{x}_i)$  |   图$\mathcal{G}$中$\mathrm{x}_i$的父节点|


##  索引



|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle a_i$  |      向量$\boldsymbol{\mathit{a}}$的第$i$个元素，其中索引从1开始|
|$\displaystyle a_{-i}$  |    除了第$i$个元素，$\boldsymbol{\mathit{a}}$的所有元素|
|$\displaystyle A_{i,j}$  |    矩阵$\boldsymbol{\mathit{A}}$的$i,j$元素|
|$\displaystyle \boldsymbol{\mathit{A}}_{i, :}$  |  矩阵$\boldsymbol{\mathit{A}}$的第$i$行|
|$\displaystyle \boldsymbol{\mathit{A}}_{:, i}$  |  矩阵$\boldsymbol{\mathit{A}}$的第$i$列|
|$\displaystyle \textit{A}_{i, j, k}$  |3维张量${\textbf{A}}$的$(i, j, k)$元素|
|$\displaystyle {\textbf{A}}_{:, :, i}$  |3维张量的2维切片|
|$\displaystyle \mathrm{a}_i$  |    随机向量$\mathbf{a}$的第$i$个元素|



##  线性代数中的操作


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle \boldsymbol{\mathit{A}}^\top$  |     矩阵$\boldsymbol{\mathit{A}}$的转置|
|$\displaystyle \boldsymbol{\mathit{A}}^+$  |      $\boldsymbol{\mathit{A}}$的的Moore-Penrose 伪逆|
|$\displaystyle \boldsymbol{\mathit{A}} \odot \boldsymbol{\mathit{B}}$  |  $\boldsymbol{\mathit{A}}$和$\boldsymbol{\mathit{B}}$的逐元素乘积（Hadamard 乘积）|
|$\displaystyle \mathrm{det}(\boldsymbol{\mathit{A}})$  |$\boldsymbol{\mathit{A}}$的行列式|


##  微积分


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle\frac{d y} {d x}$  |               $y$关于$x$的导数|
|$\displaystyle \frac{\partial y} {\partial x}$  |       $y$关于$x$的偏导|
|$\displaystyle \nabla_{\boldsymbol{\mathit{x}}} y$  |               $y$关于$\boldsymbol{\mathit{x}}$的梯度|
|$\displaystyle \nabla_{\boldsymbol{\mathit{X}}} y$  |               $y$关于$\boldsymbol{\mathit{X}}$的矩阵导数|
|$\displaystyle \nabla_{\textbf{X}} y$  |               $y$关于${\textbf{X}}$求导后的张量|
|$\displaystyle \frac{\partial f}{\partial \boldsymbol{\mathit{x}}}$  |       $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$m 的Jacobian矩阵$\boldsymbol{\mathit{J}} \in \mathbb{R}^{m\times n}$|
|$\displaystyle \nabla_{\boldsymbol{\mathit{x}}}^2 f(\boldsymbol{\mathit{x}})\text{ or }\boldsymbol{\mathit{H}}( f)(\boldsymbol{\mathit{x}})$  |$f$在点$\boldsymbol{\mathit{x}}$处的Hessian矩阵|
|$\displaystyle \int f(\boldsymbol{\mathit{x}}) d\boldsymbol{\mathit{x}}$  |              $\boldsymbol{\mathit{x}}$整个域上的定积分|
|$\displaystyle \int_\mathbb{S} f(\boldsymbol{\mathit{x}}) d\boldsymbol{\mathit{x}}$  |           集合$\mathbb{S}$上关于$\boldsymbol{\mathit{x}}$的定积分|


##  概率和信息论


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle \mathrm{a} \bot \mathrm{b}$  |                  $\mathrm{a}$和$\mathrm{b}$相互独立的随机变量|
|$\displaystyle \mathrm{a} \bot \mathrm{b} \mid \mathrm{c}$  |             给定$\mathrm{c}$后条件独立|
|$\displaystyle P(\mathrm{a})$  |                      离散变量上的概率分布|
|$\displaystyle p(\mathrm{a})$  |                      连续变量（或变量类型未指定时）上的概率分布|
|$\displaystyle \mathrm{a} \sim P$  |                    具有分布$P$的随机变量$\mathrm{a}$|
|$\displaystyle  \mathbb{E}_{\mathrm{x}\sim P} [ f(x) ]\text{ or } \mathbb{E} f(x)$  |$f(x)$关于$P(\mathrm{x})$的期望|
|$\displaystyle \text{Var}(f(x))$  |                    $f(x)$在分布$P(\mathrm{x})$下的方差|
|$\displaystyle \text{Cov}(f(x),g(x))$  |                  $f(x)$和$g(x)$在分布$P(\mathrm{x})$下的协方差|
|$\displaystyle H(\mathrm{x})$  |                      随机变量$\mathrm{x}$的的香农熵|
|$\displaystyle D_{\text{KL}} ( P \Vert Q )$  |            P和Q的的KL 散度|
|$\displaystyle \mathcal{N} ( \boldsymbol{\mathit{x}} ; \boldsymbol{\mu} , \boldsymbol{\Sigma})$  |        均值为$\boldsymbol{\mu}$协方差为$\boldsymbol{\Sigma}$，$\boldsymbol{\mathit{x}}$上的高斯分布|

##  函数


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle f: \mathbb{A} \rightarrow \mathbb{B}$  |  定义域为$\mathbb{A}$值域为$\mathbb{B}$的函数$f$|
|$\displaystyle f \circ g$  |          $f$和$g$的组合|
|$\displaystyle f(\boldsymbol{\mathit{x}} ; \boldsymbol{\theta})$  |       由$\boldsymbol{\theta}$参数化，关于$\boldsymbol{\mathit{x}}$的函数（有时为简化表示，我们忽略$\boldsymbol{\theta}$记为$f(\boldsymbol{\mathit{x}})$ ）|
|$\displaystyle \log x$  |            $x$的自然对数|
|$\displaystyle \sigma(x)$  |          Logistic sigmoid, $\displaystyle \frac{1} {1 + \exp(-x)}$|
|$\displaystyle \zeta(x)$  |           Softplus, $\log(1 + \exp(x))$|
|$\displaystyle || \boldsymbol{\mathit{x}} ||_p$  |         $\boldsymbol{\mathit{x}}$的$L^p$范数|
|$\displaystyle || \boldsymbol{\mathit{x}} ||$  |          $\boldsymbol{\mathit{x}}$的$L^2$范数|
|$\displaystyle x^+$  |             $x$的正数部分, 即$\max(0,x)$|
|$\displaystyle \textbf{1}_\mathrm{condition}$  |如果条件为真则为1，否则为0|

有时候我们使用函数$f$，它的参数是一个标量，但应用到一个向量、矩阵或张量：
$f(\boldsymbol{\mathit{x}})$, $f(\boldsymbol{\mathit{X}})$, or $f({\textbf{X}})$ 。 这表示逐元素地将$f$应用于数组。
例如，${\textbf{C}} = \sigma({\textbf{X}})$，则对于所有合法的$i$、$j$和$k$，
$\textit{C}_{i,j,k} = \sigma(\textit{X}_{i,j,k})$。


##  数据集和分布


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle p_{\text{data}}$  |      数据生成分布|
|$\displaystyle \hat{p}_{\text{train}}$  |   由训练集定义的经验分布|
|$\displaystyle \mathbb{X}$  |           训练样本的集合|
|$\displaystyle \boldsymbol{\mathit{x}}^{(i)}$  |         数据集的第$i$个样本（输入）|
|$\displaystyle y^{(i)}\text{ or }\boldsymbol{\mathit{y}}^{(i)}$  |监督学习中与$\boldsymbol{\mathit{x}}^{(i)}$关联的目标|
|$\displaystyle \boldsymbol{\mathit{X}}$  |            $m \times n$ 的矩阵，其中行$\boldsymbol{\mathit{X}}_{i,:}$为输入样本$\boldsymbol{\mathit{x}}^{(i)}$|


##  模型相关符号定义


|   符号   |   定义   |
| ---: | :--- |
|$\displaystyle J$  |      损失函数|
|$\displaystyle \boldsymbol{\mathit{W}}$  |模型参数向量或矩阵|
|$\displaystyle b$  |          模型bias标量|
|$\displaystyle \boldsymbol{\mathit{b}}$  |          模型bias向量|
