---
layout: post
title: support vector machine
date:  2017-07-03 08:00
description: the basic support vector machine
tag: machine learning
---

# Part one **训练数据在特征空间中线性可分**

## **1. Origin Problem**

假设空间中有两类数据点，要找出一个超平面(hyperplane)使得各点能正确分类，如下图:

![图1](/images/posts/machine learning/support vector machine/img1.jpg)

图中超平面表示为$w^{T}x+b = 0$，则空间中任何一个点$x$到该超平面的距离为$\frac{\vert w^{T}x+b \vert}{\Vert{w} \Vert}$

> proof: 设超平面 $S: w^{T}x+ b = 0$ ，则超平面上的点 $ x = \frac{-b}{\Vert w \Vert} $ , 空间中的任一点 $x_0$
与S的距离为 $ d=\frac{|b1-b|}{\Vert w \Vert} = \frac{| w^{T}x_0+b |}{\Vert w \Vert} $ ，其中$ b_1$为$x_0$过与S平行的超平面 $S^{\'} : w^{T}x + b_1 = 0$ 的偏移值

若有N个输入向量$X_1, X_2, ..., X_N$，分别属于类别为：$t_1, t_2, ..., t_N$，其中$t_n \in \begin{Bmatrix}-1 &, & 1\end{Bmatrix}$。
现假设训练数据在特征空间中线性可分，则对与任何一点$x_i$，满足$y(x_i) > 0 $，$t_i = +1$；$y(x_i) < 0$，$t_i = -1$，
即：$t_ny(x_n) > 0$

上面二分类问题，很多方法可以找出一个超平面分离两种数据，比如感知机(perceptron)就可以在有限步内找到一个可行解。但是怎样的超平面才是最好的呢？，支持向量机选择的是**边界上点所形成的超平面margin最大**对应的超平面！

### **1.1. 边界上的点和margin**

这里所谓的边界上的点，即离超平面S最近的点，见下图A点，margin的定义可以看如下图：

![图二](/images/posts/machine learning/support vector machine/img2.jpg)

综上，我们得出原始的目标函数:

$$
arg \quad \max_{w,b} \begin{Bmatrix}
\frac{1}{\Vert w \Vert} \min \limits_{n}[t_n(w^{T}x_n+b)]
\end{Bmatrix}\tag{1}
$$

上述这个目标函数如果直接求解会很复杂，可以考虑将$min$部分改为constraint，具体如下：

&ensp; 使最靠近超平面的任意点i满足$t_i(w^{T}x_i+b) = 1, \quad$，则所有点满足$t_n(w^{T}x_n+b) \geq 1, \quad n=1,2,...,N$。
完成这个操作仅需让目标函数分子分母同乘一个$k： w \Rightarrow kw, \quad b \Rightarrow kb$，整个式子值不变！
通过上面变换，我们的目标函数变为$arg \quad \max \limits_{w,b} \frac{1}{\Vert w \Vert}$，等价于:

$$
\begin{align*}
arg \quad \min\limits_{w,b} \frac{1}{2} \Vert w \Vert^2 \\
subject \quad to:  \\
\quad \quad \quad t_n(w^{T}x_n+b) \geq 1, \quad n=1,2,...,N
\end{align*}\tag{2}
$$

## **2. 拉格朗日对偶问题**

解决上述带constraint的优化问题，我们自然想到引入拉格朗日乘子(lagrange multipliers)$\lambda_n \geq 0$，有如下拉格朗日函数：

$$

L(w,b,\lambda) = \frac{1}{2} \Vert w \Vert^2 - \sum \limits_{n=1}^{N}\lambda_n\begin{Bmatrix} t_n(w^{T}x_n+b)-1\end{Bmatrix}\tag{3}

$$

### **2.1. 为什么拉格朗日乘子的符号为负**

&ensp; 一般我们在引入拉格朗日乘子时，其前面符号为正或者负都是可以的。例如下图，假设优化函数$f(x)$，constraint为$g(x) = 0$平面，易知$\nabla{g(x)}$为$g(x)$切平面的法线。
那么，为了求$f(x)$的最大值（最小值），$ \nabla{f(x)}$必须于$ \nabla{g(x)}$平行（否则我们可以沿着constraint平面移动小步$x+\epsilon$。
使得$f(x+\epsilon) > f(x)$ ($f(x+\epsilon) < f(x) $)。即：

$$

\nabla{f(x)} + \lambda \nabla g(x) = 0 , \quad \lambda 的符号不定

$$


![图三](/images/posts/machine learning/support vector machine/img3.jpg)

现在假设我们求目标函数$f(x)$的最大值，constraint为$g(x) \geq 0$，则对于取得最优值的点x有两种情况:

1. x在$g(x) > 0$内（即在区域内），则此时$g(x)$并没有对目标函数有作用，即$\lambda = 0$。因此相当于无条件最值问题。
2. x在$g(x) = 0$上（即在边界上），则此时$\nabla{f(x)}$方向必须远离区域$g(x) > 0$，否则根据梯度方向$\nabla{f(x)}$为函数增长最快方向，最大值在$g(x)$内取得，这与假设不符。
因此，$\nabla{f(x)} = -\lambda \nabla{g(x)}，
\lambda > 0$。如下图所示:

![图四](/images/posts/machine learning/support vector machine/img4.jpg)

### **2.2. 引出拉格朗日对偶问题**

我们回到原先的拉格朗日函数，我们求的是最小值问题，根据梯度方向$\nabla{f(x)}$为函数下降最快的方向，此时$\nabla{f(x)}$应该指向$g(x) > 0$内，
即$\nabla{f(x)} = \lambda \nabla{g(x)}，\lambda > 0 \Rightarrow \nabla{f(x)} - \lambda \nabla{g(x)} = 0，\lambda > 0$，则易推出拉格朗日函数:
$L(x,\lambda) =  f(x) - \lambda g(x)$

对$L(w,b,\lambda)$分别求$w$和$b$的偏导：

$$

\begin{align*}
w & = & \sum \limits_{n=1}^{N}{\lambda_{n}t_n{x_n}} \\
0 & = & \sum \limits_{n=1}^{N}{\lambda_{n}t_n}
\end{align*}\tag{4}

$$

将上面得到两个式子代入拉格朗日函数(3)中，消去w,b，得到其拉格朗日对偶函数：

$$

L(\lambda)^{'} = \sum \limits_{n=1}^{N}\lambda_n - \frac{1}{2} \sum \limits_{n=1}^{N} \sum \limits_{m=1}^{N}
\lambda_{n}\lambda_{m}t_{n}t_{m}k(x_n,x_m)\tag{5}

$$

其中$k(x, x^{\'})$为核函数(kernel)，我们以后会详细介绍。这里假设核函数为$k(x, x^{\'}) = x^{T}x^{\'}$。由拉格朗日对偶问题[1]知，拉格朗日对偶函数$L(\lambda)^{'}$是$f(x)$的最小值$p^{ * } $的下界，**那么什么样的下界是最好的呢？当然是值越大的下界越好**(因为只有当$L(\lambda)^{\'}$取得最大的那个值，才是对$p^{ * }$最终有效的下界，换句话说只有这样才能有效缩小$p^{ * } $的范围)。
最终原始优化问题的拉格朗日对偶函数问题(lagrange dual problem):

$$
\begin{align*}
\max L(\lambda)^{'} = \sum \limits_{n=1}^{N}\lambda_n - \frac{1}{2} \sum \limits_{n=1}^{N} \sum \limits_{m=1}^{N}
\lambda_{n}\lambda_{m}t_{n}t_{m}k(x_n,x_m) \\
subject \quad to \\

\quad \quad \lambda_n \geq 0, \quad n = 1, 2, ...,N \\
\quad \quad \sum \limits_{n=1}^{N} {t_{n}\lambda_{n}} = 0
\end{align*}\tag{6}

$$

**同时根据Slater's condition原始问题和对偶等价(感兴趣的读者可以看Convex Optimization)** 。上面是一个带条件的quadratic problem，在训练数据量比较少的时候，可以用quadratic problem solver直接解决，但是当数据量比较大时，传统的solver效率非常低。

## **3. KKT 条件**

先来看看在2.1中，我们得到的结论:

假设我们求目标函数$f(x)$的最大值(或者最小值)，constraint为$g(x) \geq 0$，则对于取得最优值的点x有两种情况:

1. x在$g(x) > 0$内（即在区域内），则此时$g(x)$并没有对目标函数有作用，即$\lambda = 0$。因此相当于无条件最值问题。
2. x在$g(x) = 0$上（即在边界上），$\lambda  > 0$
综上，得出$\lambda{g(x)} = 0$，则拉格朗日函数$L(x,\lambda) = f(x) + \lambda g(x)$的constrians如下:

$$
\begin{align*}
g(x) \geq 0       \\
\lambda \geq 0    \\
\lambda{g(x)} = 0 \\
\end{align*}\tag{7}
$$

上面的constraints称为Karush-Kuhn-Tucker(KKT)condition(Karush,1939;Kuhn and Tucker, 1951).因此可以看出
KKT条件实际上是拉格朗日函数的一些基本性质!

下面回到我们的目标函数，不难得到如下KKT条件：
$$
\begin{align*}
\lambda_n \geq 0              \\
t_{n}y_{n}(x_n) - 1 \geq 0    \\
\lambda_n{(t_{n}y(x_n) - 1)} = 0
\end{align*}\tag{8}
$$

现在假设我们已经训练出模型，则需要根据得到的$\lambda_n$和kernel函数来预测$$y(x_n)$$的符号，
将$y(x_n) = w^{T}x_n + b$，用(4)得到的条件带入。即：
$$
y(x_n) = \sum \limits_{n=1}^{N}{\lambda_{n}t_{n}k(x,x_{n})} + b \tag{9}
$$

注意上面的$x_n$与$\lambda_{n}$是一 一对应的。对于每一个点$x_n$,要么$\lambda_{n} = 0$，要么$t_{n}y(x_n) = 1$。$\lambda_n = 0$的点，其不会在(9)式中有任何作用，
因此其对于预测一个输入值$x_n$的类别，没有帮助。对于其他$t_{n}y(x_n) = 1$的点，其将会对预测$x_n$所属类别贡献自己的作用，称为**支持向量**。
实际上，这些点就是我们在前面提到的**最靠近超平面的点**。

# Part two **训练数据允许被误分**

第一部分我们假设训练数据在特征空间线性可分，这在现实应用中很难满足，因此我们适当放宽条件，引入松弛变量$\xi_{n} \geq 0, \quad n =1, 2, ..., N$,
$\xi_{n}$和$x_n$一 一对应，如下图：

![图5](/images/posts/machine learning/support vector machine/img5.jpg)


$$
\left\{
\begin{eqnarray*}
&& \xi_{n} = 0, \quad x_n正确分类           \\
&& 0 < \xi_{n} \leq 1, \quad x_n在margin内  \\
&& \xi_{n} > 1, \quad x_n在错误的一边，即被错误分类        \\
\end{eqnarray*}
\right.
$$

通过上面分析，我们将(2)中的constraint稍微修改一下: $t_{n}y(x_n) \geq 1 - \xi_n; \quad n = 1, 2, ..., N$
现在我们优化的目标是：**在允许数据被误分的情况下，margin最大**。对于被误分的数据，我们当然希望其离正确类的边界越近越好，因此
我们又称$\xi_n$为slack variable penalty，并定义如下目标函数:

$$
\begin{eqnarray*}
&& \min C\sum\limits_{n=1}^{N}\xi_{n} + \frac{1}{2}\Vert w \Vert_2 \\
&& \quad subject \quad to: \\
&& \quad \xi_n & \geq  0  \\
&& \quad t_{n}y(x_n)  \geq &1 - \xi_n
&& \quad \end{eqnarray*} \tag{10}
$$

其中C>0，其作用是保持slack variable penalty和margin大小的平衡。对应的拉格朗日函数:

$$

L(w,b,\lambda) = \frac{1}{2}\Vert w \Vert^2 + C \sum \limits_{n=1}^{N}{\xi_n} - \sum\limits_{n=1}^{N}{\lambda_{n}(t_{n}y(x_n) - 1 + \xi_n)} - \sum \limits_{n=1}^{N}\mu_{n}\xi_{n} \tag{11}

$$

其中，$\lambda_n \geq 0$，$\mu_n \geq 0$为拉格朗日乘子。则相应的KKT条件如下：

$$
\begin{equation*}
\left\{
\begin{aligned}[c]
\lambda_n & \geq & 0             \\
t_{n}y(x_n)-1+\xi_n & \geq & 0   \\
\lambda_n(t_{n}y(x_n) - 1 + \xi_n) & = & 0 \\
\mu_n & \geq & 0                 \\
\xi_n & \geq & 0                 \\   
\mu_{n}\xi_{n} & = & 0   
\end{aligned}
\right.

\qquad\Longleftrightarrow\qquad

\left\{
\begin{aligned}[c]
\lambda_n = 0 \Leftrightarrow t_{n}y_{n} \geq 1  \\
0 < \lambda_n < C \Leftrightarrow t_{n}y_{n} = 1 \\
\lambda_n = C \Leftrightarrow t_{n}y{n} \leq 1   \\
\end{aligned}
\right.

\qquad where \quad n = 1,2,3...,N
\end{equation*}\tag{12}
$$

将$y(x)$用$w，b$表示，并分别对 $w，b$ 和 $\xi$ 求偏导，得到如下条件:

$$

\begin{align*}

\frac{\nabla{L}}{\nabla{w}} =  0 & \Rightarrow & w = \sum \limits_{n=1}^{N}{\lambda_{n}t_{n}x_n}       \\
\frac{\nabla{L}}{\nabla{b}}  =  0 & \Rightarrow & 0 = \sum \limits_{n=1}^{N}{\lambda_{n}t_n}        \\
\frac{\nabla{L}}{\nabla{\xi_n}}  =  0 & \Rightarrow & \lambda_n = C - \mu_n

\end{align*}\tag{13}

$$

将(13)中的条件带入拉格朗日函数(11)中，得到拉格朗日对偶函数：

$$
L^{'}(\lambda) = \sum \limits_{n=1}^{N}\lambda_{n} - \frac{1}{2}\sum\limits_{n=1}^{N}{\sum\limits_{m=1}^{N}
\lambda_{n}\lambda_{m}t_{n}t_{m}k(x_n,x_m)} \tag{14}
$$

可以看出这个函数和(6)中的函数完全一样。又因为$\lambda_n = C - \mu_n$，且$\mu_n \geq 0$，则$0 \leq \lambda_n \leq C$，称为**box constraints**
故由此可知引入松弛变量后，拉格朗日对偶函数问题及其constraints为：

$$
\begin{align*}
\max L(\lambda)^{'} = \sum \limits_{n=1}^{N}\lambda_n - \frac{1}{2} \sum \limits_{n=1}^{N} \sum \limits_{m=1}^{N}
\lambda_{n}\lambda_{m}t_{n}t_{m}k(x_n,x_m) \\
subject \quad to \\

\quad \quad 0 \leq \lambda_n \leq C, \quad n = 1, 2, ...,N \\
\quad \quad \sum \limits_{n=1}^{N} {t_{n}\lambda_{n}} = 0
\end{align*}\tag{15}

$$

# Part Three **SMO算法**

我们知道上面的拉格朗日对偶问题，属于QP(quadratic problem)问题，标准的QP solver很难有效的解决训练数据比较大的SVM问题。因此，
近十几年来很多牛人发明了许多诸如'chunking'、'Osuna's algorithm'等方法，其都试图将庞大的QP问题，化为一些小的子问题，以便可以
有效的解决问题。 John C. Platt参考Osuna's algorithm，发明了SMO算法(Sequential Minimal optimization)是广为大家使用的方法，
接下来我们就详细介绍该方法。


## **1. SMO算法思想**

SMO算法每次迭代选择两个拉格朗日乘子参与优化更新，并得到当前最优值。之所以选择两个乘子，是因为(15)中第二个constraint的约束，使得
最小的拉格朗日函数子问题为两个拉格朗日乘子优化问题!
那么，我们自然而然的想到两个问题:

1. 如何选择本次参与优化的两个拉格朗日乘子
2. 如何解决两个拉格朗日乘子的优化问题

先来介绍如何解决两个拉格朗日乘子优化问题

## **2. 两个拉格朗日乘子的优化问题**

根据(15)中的两个约束条件，两个拉格朗日乘子$\lambda_1$，$\lambda_2$的关系见如下图:

![图6](/images/posts/machine learning/support vector machine/img6.JPG)
其中左图对应$y_1 \neq y_2$，右图对应$y_1 = y_2$。则若$0 \leq \lambda_1 \leq C$，根据上面的关系我们可以得到$\lambda_2$的范围：

$$
\begin{align*}

L & = & \max(0,\lambda_2 - \lambda_1)， H  =  \min(C, C+\lambda_2 - \lambda_1) \quad y_1 = y_2 \\
L & = & \max(0, \lambda_1 + \lambda_2 - C)， H = \min(C, \lambda_1 + \lambda_2) \quad y_1 \neq y_2

\end{align*}\tag{16}
$$

将上面的拉格朗日函数写成$\lambda_1$，$\lambda_2$的形式：

$$
\begin{eqnarray*}
&& \Psi = \lambda_{1} + \lambda_{2}-(\frac{1}{2}K_{11}\lambda_{1}^{2} + \frac{1}{2}K_{22}\lambda_{2}^{2} + sK_{12}\lambda_{1}\lambda_{2} + t_{1}\lambda_{1}v_{1} +
t_{2}\lambda_2\lambda_{2}v_{2})  - \Psi_{constant} \\
where \\
&& K_{ij}  =  K(x_{i},x_{j}) \\
&& v_{i} = \sum \limits_{j=3}^{N}t_{i}\lambda_{j}^{ * }K_{ij} = y_{i} - b^{ * }  - t_1\lambda_{1}^{ * }K_{1i} - t_2\lambda_{2}^{ * }K_{2i}
\end{eqnarray*}\tag{17}
$$
其中$s = t_{1}t_{2}，\Psi_{constant}为与\lambda_1,\lambda_2无关的部分$，**\***代表前一次迭代更新值。根据$\lambda_1$和$\lambda_2$之间的关系有：

$$
\lambda_1 + s\lambda_2 = \lambda_{1}^{ * } + s\lambda_{2}^{ * } = \Gamma \tag{18}
$$

带入(16)中，得到:

$$

\Psi(\lambda_2) = \Gamma - s\lambda_2 + \lambda_2 -(\frac{1}{2}k_{11}(\Gamma-s\lambda_2)^2 + \frac{1}{2}k_{22}{\lambda_{2}^{2}}+sk_{12}(\Gamma-s\lambda_2)\lambda_2
+t_{1}(\Gamma-s\lambda_2)v_{1} + t_{2}\lambda_{2}v_{2}) - \Psi_{constant} \tag{19}

$$

对$\lambda_2$求导,并令其为0得到:

$$
\lambda_2(k_{11} + k_{22} - 2k_{12}) = s(k_{11} - k_{12})\Gamma + t_2(v_1 - v_2) + 1 - s \tag{20}
$$

再根据 $\Gamma = \lambda_{1}^{ * } + s\lambda_{2}^{ * }$和$v_1$，$v_2$的值，带入(20)中，得到：

$$
\lambda_{2}(k_{11}+k_{22}-2k_{12}) = \lambda_{2}^{ * }(k_{11} + k_{22} - 2k_{12}) +t_2(y_1 - y_2 + t_2 - t_1) \tag{21}
$$

综上，我们终于得到$\lambda_{2}$的更新公式（(●ˇ∀ˇ●)）：

$$
\begin{eqnarray*}
&& \lambda_{2}^{new} = \lambda_{2} + \frac{t_{2}(E_1 - E_2)}{k_{11} + k_{22} - 2k_{12}} \\
&& where \\
&& E_i = y_i - t_i \\
\end{eqnarray*}\tag{22}
$$


好开心有木有，终于快Over了，是不是想撸起代码来实现了。然而，别急！$\lambda_2$更新后可能会不在原先的范围内，
因此很有必要修正：

$$

\lambda_{2}^{new, clipped} = \left\{
\begin{eqnarray*}
&&    H, \quad if \quad \lambda_{2}^{new} \geq H  \\
&&    \lambda_{2}^{new}, \quad if \quad L < \lambda_{2}^{new} < H \\
&&    L, \quad if \quad \lambda_{2}^{new} \leq L \\    
\end{eqnarray*}
\right.\tag{23}
$$

则$\lambda_{1}^{new} = \lambda_{1} + s(\lambda_2 - \lambda_{2}^{new,clipped})$，易知$\lambda_{1}$无需修正。
<(＿　＿)>，总算完了。咦是不是忘记啥了!。假设我们每次$\lambda_1$和$\lambda_2$更新好了，那么每次选择的训练数据 $x_{i}$ ，**用于预测 $y_{i}$ 的 b** 是不是根据(9)也要更新呢？这是肯定的。则对于$\lambda_1$，我们有：

$$
\begin{align*}
& t_1(\sum \limits_{m \in  S}{\lambda_{m}t_{m}K_{nm}+b}) = 1                        \\
& \Longrightarrow \sum \limits_{m \in  S}{\lambda_{m}t_{m}K_{nm}}+b_1 = t_{1}       \\
& \Longrightarrow b  = t_{1} - \sum \limits_{m \in  S}{\lambda_{m}t_{m}K_{nm}}      \\
& \Longrightarrow b^{new}_{1} = b - t_{1}(\lambda_{1}^{new} - \lambda_{1})K_{11} - t_{2}(\lambda_{2}^{new,clipped} -  \lambda_{2})
\end{align*}\tag{24}
$$

理论上每一个支持向量满足：$t_{n}y(x_{n}) = 1$，但是在计算机计算的过程中，我们需要容许误差，例如当$t_{n}y(x_{n}) = 0.999$，
$t_{n}y(x_{n}) = 1.001$也认为正确的。因此上面的 $b^{new}$ 需要减去$E_{1}$来修正：

$$
b^{new}_{1} = b - E_{1} - t_{1}(\lambda_{1}^{new} - \lambda_{1})K_{11} - t_{2}(\lambda_{2}^{new,clipped} -  \lambda_{2})K_{12} \tag{25}
$$

同理：

$$
b^{new}_{2} = b - E_{2} - t_{2}(\lambda_{1}^{new} - \lambda_{1})K_{12} - t_{2}(\lambda_{2}^{new,clipped} -  \lambda_{2})K_{22} \tag{26}
$$


至此SMO的最核心的部分已经解释完了，我们详细介绍了 **两个拉格朗日乘子每一轮迭代中如何优化并更新相关的参数** 。

最后还有一个程序上的问题：如何选择两个拉格朗日乘子来更新。

## **3. 两个拉格朗日乘子的优化问题**

Platt论文[4]中2.2部分已经详细介绍了两个乘子的更新部分，不过要结合伪代码才能正确理解。首先要明确的是
两个乘子第一个乘子是在外层循环更新（可理解为**主乘子**)，第二个乘子根据**max step size**最大来选择，这里**max step size**即：$$|E1 - E2| $$。
**Osuna的理论中有证明：若至少有一个拉个拉格朗日乘子违反KKT条件，则每一个更新乘子都能使目标函数(文中将目标函数写成max f(X)，本质是一样的)减少。因此我们不必担心选择的乘子使目标函数不收敛!!!**

下面介绍一下两个乘子的具体选择步骤：

+ P1. 第一个乘子选择更新按照如下规则：

&ensp;&ensp;    **A.** 第一次更新时，遍历所有的训练数据，并判断每个数据是否违反KKT条件（若违反说明可以优化），并更新相关的$\lambda$值。

&ensp;&ensp;    **B.** 接着遍历所有non-bound的训练数据，并像A中一样对其判断是否违反KKT条件，并更新相关的$\lambda$值，直到所有的non-bound数据满足KKT条件,则返回A步骤

&ensp;&ensp;    **C.** 若迭代次数用完**或者**所有的训练数据满足KKT条件

+ P2. 第二个乘子更新按照如下规则：

&ensp;&ensp;    **A.** 每次选1中遍历一个乘子$\lambda_i$后，会更新一个全局的eCache，eCache[i]表示第i个数据的$E_i$，$eCache[i]$有两个数据，第一个表示
                   这个$E_i$是否有效，第二个记录$E_i$值。$\lambda_i$若其违反KKT条件则会接着选择第二个乘子。

&ensp;&ensp;    **B.** 第二个乘子的选择即根据所有有效的$E_j$值，找出与$E_i$相差最大的$E_j$对应的数据更新相关数据


下面我摘录了<machine learning in action>中的代码如下，并对关键部分做了注释，现在大家看到代码应该能理解所有的tricks：

~~~
# svmCore.py

from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

    class optStruct:
        def __init__(self, dataMatIn, classLabels, C, toler):
            self.X = dataMatIn
            self.labelMat = classLabels
            self.C = C
            self.toler = toler
            self.m = shape(dataMatIn)[0]
            self.alphas = mat(zeros((self.m, 1)))
            self.b = 0
            self.eCache = mat(zeros((self.m, 2)))

    def calcEk(oS, k):
        fXK = float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
        Ek = fXK - float(oS.labelMat[k])
        return Ek

    def selectJ(i, oS, Ei):
        maxK = -1; maxDeltaE = 0; Ej = 0
        oS.eCache[i] = [1,Ei]
        validEcacheList = nonzero(oS.eCache[:,0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue;
                Ek = calcEk(oS,k)
                deltaE = abs(Ei- Ek)
                if deltaE > maxDeltaE:
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else: # 没有有效的eCache，只能随机选择一个J
            j = selectJrand(i, oS.m)
            Ej = calcEk(oS,j)
        return j, Ej

    def updateEk(oS, k):
        Ek = calcEk(oS,k)
        oS.eCache[k] = [1,Ek]

    def innerL(i,oS):
        Ei = calcEk(oS,i)
        if (oS.labelMat[i]*Ei < -oS.toler and oS.alphas[i] < oS.C) or (oS.labelMat[i]*Ei > oS.toler and oS.alphas[i] > 0): # 测试KKT条件
            j,Ej = selectJ(i, oS, Ei)
            alphaIOld = oS.alphas[i].copy(); alphaJOld = oS.alphas[j].copy()
            if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
                H = min(oS.C, oS.alphas[i] + oS.alphas[j])
            if L == H:
                print("L == H")
                return 0
            eta = oS.X[i,:]*oS.X[i,:].T + oS.X[j,:]*oS.X[j,:].T - 2*oS.X[i,:]*oS.X[j,:].T
            if eta <= 0:
                print('eta <= 0')
                return 0
            oS.alphas[j] += oS.labelMat[j]*(Ei - Ej) / eta
            oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
            updateEk(oS,j) # 这里按论文中的描述，应该是等alphas[i]更新后一起更新，注意一下！！！
            if abs(oS.alphas[j] - alphaJOld) < 0.00001:
                print('j not moving enough')
                return 0
            oS.alphas[i] += oS.labelMat[i]*oS.labelMat[j]*(alphaJOld - oS.alphas[j])
            updateEk(oS,i)
            # updateEk(oS,j) 应该放这里 !!!
            b1 = oS.b - Ei -oS.labelMat[i]*(oS.alphas[i] - alphaIOld)*oS.X[i,:]*oS.X[i,:].T \
                -oS.labelMat[j]*(oS.alphas[j] - alphaJOld)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej -oS.labelMat[i]*(oS.alphas[i] - alphaIOld)*oS.X[i,:]*oS.X[j,:].T \
                -oS.labelMat[j]*(oS.alphas[j] - alphaJOld)*oS.X[j,:]*oS.X[j,:].T
            if 0 < oS.alphas[i] and oS.alphas[i] < oS.C:
                oS.b = b1
            elif 0 < oS.alphas[j] and oS.alphas[j] < oS.C:
                oS.b = b2
            else:
                oS.b = (b1+b2)/2.0
            return 1
        else:
            return 0

    def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
        oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet: # only when alphaPairsChanged == 0, it is activated
                for i in range(oS.m):
                    alphaPairsChanged += innerL(i,oS)
                    print( 'fullSet, iter: {0}, i : {1}, pairs changed {2}'.format(iter, i, alphaPairsChanged))
                iter += 1
            else:
                nonBoundIs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < oS.C))[0] # wow!!!
                for i in nonBoundIs:
                    alphaPairsChanged += innerL(i, oS)
                    print('non-bound, iter: {0}, i : {1}, pairs changed {2}'.format(iter,i,alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            print('iteration number: {0}'.format(iter))
        return oS.b, oS.alphas

    def calcW(alphas, dataArr, classLabels):
        X = mat(dataArr);labelMat = mat(classLabels).transpose()
        m, n = shape(X)
        w = zeros((n,1))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i], X[i,:].T)
        return w    

~~~

上面``` updateEk(oS,j) ``` 按照Platt[4]的伪代码应该和 ``` updateEk(oS,i) ``` 一起，不知Petter为何放这里，大家注意一下!!
其中，```loadDataSet(fileName)``` 函数是加载数据，数据可以从Peter给的源码中得到，下面给出主函数：

~~~
# main.py

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import svmCore
dataArr, labelArr = svmCore.loadDataSet('testSet.txt')

b, alphas = svmCore.smoP(dataArr, labelArr, 0.6, 0.001, 40)
ws = svmCore.calcW(alphas, dataArr, labelArr)
print(ws)

m,n = shape(alphas)
suppVec = []
suppVecLabel = []
suppAlpahs = []
for i in range(m):
    if alphas[i] > 0.0: suppVec.append(dataArr[i])

'''
Plot the support vectors
'''
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers = []
colors = []
fr = open('testSet.txt')
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if label == -1:
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)
fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker='s', s = 90)
ax.scatter(xcord1, ycord1, marker='o', s = 50, c = 'red')
plt.title('Support Vectors Circled')

for p in suppVec:
    circle = Circle((p[0], p[1]), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
x = arange(-2.0, 12.0, 0.1)
y = (-ws[0]*x - float(b)) / ws[1]
ax.plot(x,y)
ax.axis([-2,12,-8,6])
plt.show()

~~~

该函数主要画出了二维数据的分离超平面。如下：

![图7](/images/posts/machine learning/support vector machine/img7.jpg)

图中圈圈表示的是支持向量，有人可能会问为什么支持向量不在最边界上靠近的点上选择呢? 这是因为我们上面讲的 $C$ 的原因，它的作用是既要保证margin尽量大，
又要保证数据分类尽量正确。本例子数据完全可以线性分离，因此可能会看上去支持向量选择的并不好。**当数据无法完全线性分离时， C 的作用才会明显，它将使
支持向量尽量靠近分离超平面!**

## **总结**

上面我们从SVM原问题逐步探讨了它等价的拉格朗日对偶问题，并且详细介绍了SMO算法。但是，其实还有两个问题没有深入：

1. 核函数Kernel：核函数的核心思想是将低维的线性难分的特征空间，升到高维的线性易分的特征空间，如高斯核:

    $$
    k(x,y) = e^{\frac{- \Vert x - y \Vert^2}{2\sigma^2}}
    $$

    核函数在很多模型中都会用到，因此本人将根据自己学习总结单独开一章来详细介绍。

2. 如何分两类以上的数据： SVM在原问题引入的过程中就一直以二分类为前提，因此基本的SVM是只能二分类的。
  C.W. Hus 的'A Comparsion of Methods for Multiclass Support Vector Machines'介绍了多个类SVM。后面我有时间也会另外介绍。

  当然， 还有很多针对SMO的优化，例如libsvm库估计里面有很多优化措施，有时间会接触一下。


**参考资料：**

1. Stephen Boyd, Lieven Vandenberghe. convex optimization, p223-224.
2. Peter Harrington. Machine Learning In Action p104-128.
3. Chritopher M. Bishop, Pattern Recognition and Machine Learning(Spring, 2006), p326-338, p707-710.
4. Jhon C.Platt, Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.
