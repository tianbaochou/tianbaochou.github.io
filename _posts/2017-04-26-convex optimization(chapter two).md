---
layout: post
title: (2)Theory - Convex Sets
date: 2017-04-26 23:21  
description: Convex Set相关理论部分,内容繁多  
tag: convex optimization
---

## **1.Introduction**

&ensp; Convex Set部分知识对本人来说挑战蛮大的，原因归于以下几点:  

+ 其中涉及许多Algebra analysis的术语，本人没有学习过
+ Boyd在很多地方一笔带过许多概念，让你不知其所以然

&ensp; 因此在学习本章的同时，我也将学习Algebra analysis的基础知识，并对Elementary linear algebra 相关概念重新复习。接下来先摸清本章的脉络，这章名为Convex Sets，就必定介绍集合的相关理论，其主要介绍了以下几个概念:  

1. Affline(仿射)

2. Convex(凸)

3. Cone(锥)

4. Hyperplane and halfspace(超平面和半空间)

5. Euclidean balls and ellipsoid(欧几里得球体和椭球)

6. Polyhedra(多面体)

7. Positive semidefinite cone(半正定锥)

8. Linear-fractional and perspective function(线性分部和透视函数)

9. Generalized inequalities(广义不等式)

10. Separating and supporting hyperplane(分离支持超平面)

11. Dual cones and generalized inequalities(对偶锥和广义对偶不等式)

&ensp; 分类方面:  

+ Part1：1~2介绍了Affline和Convex的定义和一些简单的例子

+ Part2：3~7为Convex Set常用的例子

+ Part3：8列举了一些保留集合凸性的一些运算

+ Part4：9介绍了广义不等式

+ Part5：10引入分离和支持超平面

+ Part6：11引入对偶堆与广义不等式概念

## **2.Theory Convex Sets**

### **Part 1 : Affline and Convex Sets**

**affline(仿射)**

定义：对于任意$x_1， x_2 \in R为R^n$的两个点，则$y = \theta x_1 + (1-\theta)x_2$  

+ 当$\theta \in(0,1)$时表示直线段

+ 当$\theta \in R$时表示直线


![图2-1](/img/posts/convex optimization/chapter two/chapter_two_1.PNG)

**仿射集(affline set)**

&ensp; 定义：

$$
\forall x_1, x_2 \in C,\quad \theta \in R \quad \theta x_1+(1-\theta)x_2 \in C
$$

&ensp; 从几何角度来看，就是过C中任意不同两个点的直线在集合C中，换句话说：任何C中的两个点，只要相关系数之和为1，
且其线性组合（或者仿射组合）包含在C中，**则C为仿射集**

&ensp; 推广到多个点：

$$
\forall x_1, x_2, ...,x_n \in C, \quad \theta_1 + \theta_2 + ... \theta_n = 1, \quad \theta_1x_1+\theta_2x_2+...+\theta_nx_n \in C
$$

则C为仿射集

**仿射包(affline hull)**

&ensp; 一个集合C的仿射包为C中所有点的仿射组合 aff C:  

$$
aff C = \{\theta_1x_1 + \theta_2x_2 + ... + \theta_kx_k\mid x_1,...,x_k \in C, \quad \theta_1+...+\theta_n=1\}
$$

易知仿射包是包含C的仿射集中最小的仿射集 $\Rightarrow$ 若S为仿射集且$C \subseteq S$，则$aff C \subseteq S$

**仿射维度(affline dimension)**

&ensp; 关于维度有一点要说明就是对于**解空间维度问题**，一般的原则是：每一个独立等式的引入，对于等式系统来说其
解的维度都将减1，此原则不仅适用于线性等式也适用于一般的代数等式。下面举个例子：  
&ensp; 在二维平面$R^2$上的单位圆：$ x_1^2 + x_2^2 = 1 $，圆的方程是一个等式，$R^2$的维度是2，因此解空间的维度为2-1。
而其仿射包为整个$R_2$平面，故维度为2。由此可以知道**_仿射维度和一般意义上的维度是有区别的。_**

**相对内部(relative interior of set C)**

&ensp; **relint C** : 包含了所有不在包含C的子空间边上的点，即：$$ relint C = \{x \in C \mid B(x,r) \cap aff C \subseteq C \quad for \  some \quad r > 0\} $$

&ensp; 这么说可能太抽象，下面举一个书上的例子:
Consider a square in the $(x_1, x_2)$-plane in $R^3$,define as:
$$
C = \{ x \in R^3 \mid -1 \leq x_1 \leq 1, \quad -1 \leq x_2 \leq 1, x_3 = 0 \}
$$，Its affline hull is the$(x_1, x_2)$-plane, i.e. $$aff C = \{x \in R^3 \mid x_3 = 0\}$$。 The interior of C
is **empty**， but the relative interior is:

$$

relint C = \{x \in R^3 \mid -1 < x_1 < 1, -1 < x_2 < 1, x_3 = 0\}

$$

&ensp; 上面interior为空是因为一个集合有非空内点的条件是: 其有包含不在边界上的点，而C集合所有的点都是在$x_3 = 0$
这个边界上，故内点为空，而从relative interior定义可以看出，其中的点只要不在任何一个边界上上即可。
因此不难看出relative interior在处理低维集合被放在高维空间时能够保留该集合在二维下的性质。

#### **2.2 凸集(convex set)**

&ensp;定义:

$$
\forall x_1, x_2 \in C, \quad  0 \leq \theta \leq 1  , \quad \theta x_1 + (1-\theta)x_2 \in C
$$

&ensp; 为了直观理解选出文中画出的典型的凸集和非凸集的例子:
图中（1）为凸集，因为其中任何两个点连接的线段上的点都在C中，而（2）为非凸集，因为图中两个黑点的连接的线段中有一部分点不在C中。

![图2-2](/img/posts/convex optimization/chapter two/chapter_two_2.PNG)


**凸包(convex hull)**

$$conv C = \{  \theta_1x_1 + ... + \theta_kx_k \mid x_i \in C, \theta_i \geq 0, \quad i = 1,2,..., k, \quad \theta_1+\theta_2+...+\theta_k = 1 \} $$
从定义可以看出C的凸包为包含C的最小凸集。

#### **2.3 锥(cones or linear cones)**

$ \forall x\in C, \theta \geq 0, \theta x \in C $则C为锥或者线性锥


**凸锥(convex cones)**

$$
\forall x_1， x_2 \in C \quad \theta_1,，\theta_2 \geq 0， \quad \theta_1x_1 + \theta_2x_2 \in C
$$
，则C为凸锥，从几何上直观理解如下图:

![图2-3](/img/posts/convex optimization/chapter two/chapter_two_3.PNG)

锥是无限延伸的没有边界，为了区别锥和凸锥，下面举个例子:

![图2-4](/img/posts/convex optimization/chapter two/chapter_two_4.PNG)

上图中$C_1 : y = \mid x \mid$为锥，但是不是凸锥，因为对于左边图像的一个点$x_1$和右边图像的点$x_2$之间组成的区域不在$C_1$中,
而灰色区域$C_2 : y \geq \mid x \mid$为凸锥。

**锥包(cones hull)**

$$
\{\theta_1x_1 + ... + \theta_kx_k \mid x_i \in C, \theta_i \geq 0, i = 1,2,..., k\}
$$
即为集合内所有的锥组合 **(非负线性组合)**，换句话说就是包含C的最小凸锥。

### **Part 2 : Some Important Examples**

**Some Simple Examples**

+ 空集、单点集、全集$R^n$均为仿射集(the empty set $\phi$，any single point(i.e. sigleton)$\{x_0\}$ and the whole space $R^n$ are affline(hence, convex) subset of $R^n$)

+ 任何直线都是仿射，若直线通过原点，则其为一个子空间，也同时为凸锥（Todo: 由凸锥定义其一定包含原点）
Any line is affline. If it passes through zeros, it is a subspace, hence also a convex cone.

+ 线段为凸但不为仿射：A line segment is convex, but not affline(unless is reduces to a point).

+ 射线$\{x_0+\theta v \mid \theta \geq 0 \}$为凸，但不是仿射，当$x_0 = 0$时为凸锥
: A ray, which has the form $\{x_0 + \theta v \mid \theta \geq 0 \}$, where $v \neq 0 $ is convex, but not affline. It is a convex cone if its base $x_0$ is 0.

+ 任何子空间都为仿射（满足数乘封闭），并为凸锥（同时满足数乘与加法封闭）: Any subspace is affline, and convex cone(hence convex)

**Another Important Examples**

#### **2.4 超平面与半空间(hyperplane and halfspace)**

**超平面**

&ensp; 定义:  $$ \{ x \mid a^Tx = b \}， \quad a \neq 0， b \in R $$,
Geometrically： $$ \{x \mid a^T(x-x_0) = 0 \} $$， 令$ a^Tx_0 = b $即可，见下图:

![图2-5](/img/posts/convex optimization/chapter two/chapter_two_5.PNG)

则超平面分割的两个半空间分别为：

$$
S1: \{x \mid a^Tx \leq b \}, \quad
S2: \{x \mid a^Tx \geq b \}
$$

&ensp; 其中$S_1$在a的反方向，$S_2$在a的方向  

&ensp; 这里简单证明一下：$S_1$中的任意一点$x_2$与a的夹角不小于90度，因此$a^T(x_2-x_0) \leq 0 &ensp;  \Rightarrow  &ensp; a^Tx_x \leq b$得证。

半开区间为$$ \{ x \mid a^T < b \} $$和$$ \{ x \mid a^T > b \}$$

#### **2.5 欧几里得球体与椭球(Euclidean balls and ellipsoid)**

**Euclidean ball in $R^n$:**

$$
B(x_c,r) = \{x \mid \parallel x - x_c \parallel \leq r \} = \{ x \mid (x - x_c)^T(x- x_c) \leq r^2 \} \\
r > 0, \parallel u \parallel_2 = (u^Tu)^\frac{1}{2}
$$

**Ellipsoid:**

$$

\varepsilon = \{ x \mid (x-c)^T P^{-1}(x-x_c) \leq 1

$$

其中$P = P^T \succ 0$ 为对称正定矩阵，$\varepsilon$的半轴长为$\sqrt{\lambda_i}， \lambda_i$为P的特征值。

**(范数锥)Norm cone**

$C = \{ (x,t) \mid \parallel x \parallel \leq t \} \subseteq R^{n+1} $

上面的概念有点抽象，下面举个例子：

&ensp; 在欧几里德范数中二阶锥(Second-order cone)或称二次锥（quadratic cone)、洛伦茨锥(Lorentz cone)是一个范数锥：

$$
\begin{aligned}
C &= \{(x,t) \in R^{n+1} \mid \parallel x \parallel_2 \leq t \} \\
  &= \left\{\begin{bmatrix}
   x \\
   t
   \end{bmatrix} \mid \begin{bmatrix} x \\ t \end{bmatrix}^T \begin{bmatrix}
   I & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ t \end{bmatrix} \leq 0, t \geq 0
  \}\right\}
\end{aligned}
$$

![图2-6](/img/posts/convex optimization/chapter two/chapter_two_6.PNG)

#### **2.6 多面体(Polyhedron)**

&ensp; 定义:

$$

P = \{ x \mid a_j^T \leq b; \quad j = 1,...,m, c_j^Tx = d_j, \quad j = 1,2,...p \}

$$

从上面可以看出，多面体为有限个半空间与超平面的相交组成的部分,见下图:

![图2-7](/img/posts/convex optimization/chapter two/chapter_two_7.PNG)


仿射集、射线、线段、半空间等都是多面体。

将上面的记法简化为：

$$

p = \{x \mid Ax \leq b, cx = d \}, \quad
A = \begin{bmatrix}
a_1^T \\
\vdots \\
a_m^T
\end{bmatrix}, \quad
C = \begin{bmatrix}
c_1^T \\
\vdots \\
c_p^T
\end{bmatrix}

$$

**Simplexes**

&ensp; 假设k+1个点，$v_0, v_1, v_2, \cdots, v_k \in R^n$为仿射不相关，即：$(v_1-v_0), (v_2 - v_0),
\cdots, (v_k - v_o)$向量两两不共线（或者不线性相关），则单形定义为：

$$

C = conv \{v_0, v_1, \cdots , v_k \} = \{ \theta_0v_0 + \cdots + \theta_kv_k \mid \theta \geq 0, 1^T\theta = 1 \}

$$

&ensp; 易知，单形的仿射维度为k，或者称在$R^n$上的k-dimensional 单形。

根据上面向量两两不共线的角度来看，1维单形为线段(line segment)，二维单形为三角形(triangle)，三维单形为四面体(tetrahedron)。

**Unit simplex: n-dimensional**

$$ 0, e_1, \cdots, e_n \in R^n : x \geq 0, 1^T \leq 1$$

**Probability simplex: (n-1)-dimensional**

由$e_1, e_2, \cdots, e_n \in R^n$决定，$x \geq 0, 1^Tx = 1$

**Simplex和Polyhedron之间的关系**

&ensp; 将Simplexs和Polyhedron放在一起是因为Simplex可以转化为Polyhedron的表示形式：

假设Simplexs为C，由Simplexs定义知 $x \in C$，当且仅当$ x = \theta_0v_0 + \theta_1v_1 + \cdots + \theta_kv_k, \quad \theta \succeq 0， 1^T\theta = 1$   
，不妨令$y = (\theta_1, \theta_2, \cdots, \theta_k)， B = [v_1 - v_0, v_2-v_0, \cdots, v_k - v_o] \in R^{n\times k}$，则$x \in C $，当且仅当：

$$
x = v_0 + By, \quad y \succeq 0， 1^T \leq 1
$$

&ensp;因为$v_0, v_1, \cdots, v_k$仿射不相关，故B的秩(rank)为k，则存在一个非奇异矩阵$A = (A_1,A_2)\in R^{n \times n}$ 使得：

$$
AB = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} B = \begin{bmatrix} I \\ 0 \end{bmatrix}
$$,
对$x = v_0 + By$ 两边左乘A，则$$\begin{bmatrix} A_1 \\ A_2 \end{bmatrix}x
= \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}v_0 + \begin{bmatrix} I \\ 0 \end{bmatrix}y$$

即：

$$
\left\{\begin{aligned}
A_1x &= A_1v_0 + y \\
A_2x &= A_2v_0
\end{aligned}
\Rightarrow
\right.
若x \in C，当且仅当
\left\{\begin{aligned}
A_2x &= A_2v_0 \\
A_1x &\succeq A_1v_0 \\
1^TA_1x &\leq 1 + 1^TA_1v_0
\end{aligned}\right.
$$

#### **2.7 半正定锥**

定义 $S^n : n\times n $为对称矩阵；$S_+^n: n \times n $为半正定对称矩阵； $\quad S_{++}^n : n \times n$为正定矩阵，其中$S_+^n$为convex cone，proof:

$$
A,B \in S_+^n, \quad x^T(\theta_1A+\theta_2B)x = \theta_1x^TAx + \theta_2x^TBx \geq 0 ,\quad \theta_1 \geq 0, \theta_2 \geq 0
$$  

### **Part 3 : 保留集合凸性的一些运算(Operations that preserve convexity)**

1. $交(Intersection)\cap： S_1为凸，S2为凸，则S_1 \cap S_2为凸.$

2. 仿射函数: $S \subseteq R^n$为凸，则$$f(S) = \{f(x) \mid x \in S\}$$ 为凸，$$f^{-1}(S) = \{x \mid f(x) \in S \}$$为凸

3. Sum: $S_1$为凸，$S_2$为凸，则$S_1 + S_2$为凸，其中$$S_1 + S_2 = \{ x + y \mid x \in S_1, y \in S_2 \}$$

4. 笛卡尔乘积(Cartesian product) : $$S_1 \times S_2 = \{(x_1, x_2) \mid x_1 \in S_1, x_2 \in S_2 \}$$为凸

5. 部分积(Partial sum of $S_1, S_2 \in R^n \times R^m$) : $$S = \{(x,y_1+y_2) \mid (x,y_1) \in S_1, \ (x,y_2)\in S_2为凸，x\in R^n ， y_i\in R^m \}$$为凸

#### **2.8 线性分部与透视函数(Linear-fractional and perspective function)**

**The perspective**
透视函数的定义如下:  

$P : R^{n+1} \rightarrow R^n, \quad domP = R^N \times R_{++}$ as $ P(z,t) = z / t$

举个简单的例子，小孔成像：

![图2-8](/img/posts/convex optimization/chapter two/chapter_two_8.PNG)

透视函数满足: $$C \subseteq dom P 为凸，则其映像 P(C) = \{P(x) \mid x \in C \}$$为Convex Image。


**Linear-fractional function**

&ensp; 假设$g : R^n \rightarrow R^{m+1}$为仿射,$$g(x) = \begin{bmatrix} A \\ C^T \end{bmatrix}x + \begin{bmatrix}
b \\ d \end{bmatrix}, \quad A \in R^{m \times n}, b \in R^m, C \in R^n , d \in R$$  
则$f: R^n \rightarrow R^m$为$f = P\circ g$，即：$$ f(x) = (Ax + b)/(C^Tx + d),\quad  domf = \{x \mid C^Tx + d > 0\} $$
称$f$为 **线性分部(Linear-fractional)或者投影(projective)**。由定义易知，当$c=0，d \geq 0$时，$domf = R^n$，$f$为一个仿射函数。    

&ensp; $$ \forall x \in C, C^Tx + d > 0，则image f(C)$$为Convex，同理$$C \subseteq R^n $$为凸，$f^{-1}(c)为Convex$

&ensp; 下面举个满足线性分部的例子：

$$
假设u,v分别在\{1,2,\cdots,n\},\quad \{1,2,\cdots,m\}中取值; P_{ij} = prob(u=i,v=j),\quad f_{ij} = prob(u=i \mid v = j) = \frac{p{ij}}{\sum_{k=1}^{n}p_{kj}}
$$
则$f$为线性分部，且若$C$为Convex set of joint probability for$(u, v)$, then the associate set of conditional probability of u given v is also convex。

### **Part 4 : 2.9. 广义不等式(Generalized inequalities)**

**真锥(Proper cone)**

&ensp;$A cone K \subseteq R^n$满足以下条件:  

1. K is convex

2. K is closed.

3. K is solid：即由非空内点

4. K is pointed： 即no line(or equivalently, $x \in K, -x \in K \Rightarrow x = 0$)

> 注：像3，4点就排除了空集和全集(如$R^2$)

为proper cone。

**广义不等式(Generalized inequalities)**

K为真锥Proper cone, 则associate with the proper cone K the patial order on $R^n$:  

$ x \preceq_ky \quad \Leftrightarrow \quad y -x \in K $

An associated strict partial order by :  

$x \prec_k y \quad \Leftrightarrow \quad y-x \in int K$

由定义易知，当$K = R_+$时， partial order $\preceq_k$ is the ususal order $\leq$ on R

**广义不等式的一些性质**

1. $\preceq_k$ is preserved under addition: $x \preceq_k y, \quad u $， then $x+u \preceq_k y+v$

2. $\preceq_k$ is transitive : $x \preceq_k y$ and  $y \preceq_k z$, then $x \preceq_k z$

3. $\preceq_k$ is preserved under nonnegative scaling: $ x \preceq_k y$ and $\alpha \geq 0$, then $\alpha x \preceq_k \alpha y$

4. $\preceq_k$ is reflexive： $x \preceq_k x$（自反性）

5. $\preceq_k$ is antisymmetric：$x \preceq_k y $ and $y \preceq_k x$，then $x = y$(反对称性)

6. $\preceq_k$ is preserved under limits：$x_i \preceq_k y_i$ for $i = 1,2,\cdots$，$x_i \rightarrow x$ and $y_i \rightarrow y$
as $i \rightarrow \infty$，then $x \preceq_k y$

**Minimum and minimal element**

**Minimum** : 最小元(偏序集中所有点都比它小，仅有一个)

A point $x \in S$是S的最小值当且仅当：$S \subseteq K + x$，这里$x+K$ **表示所有点都可以** 与$x$相比较且大于等于$x$

**Mimimal** : 极小元（偏序集中要么点小于等于它，要么无法与它比较，可能有多个）

A point $x \in S$是S中的极小值当且仅当：$(x-K)\cap = \{x\}$，这里$x-K$ **表示所有可以** 与$x$相比较且小于等于$x$的点

下图分别代表了最小元与极小元：

![图2-9](/img/posts/convex optimization/chapter two/chapter_two_9.PNG)

&ensp; 为了区分这两个概念的区别，我们举个例子：

&ensp; 对于中心在原点的椭圆，可以用一个矩阵$A \in S_{++}^n$与之关联: $$ \varepsilon_A = \{x \mid x^T A^{-1} \leq 1 \} $$
若$A \preceq B$当且仅当$\varepsilon_A \subseteq \varepsilon_B$。
给定$v_1, \cdots, v_k \in R^n$，定义：
$$
S = \{P \in S_{++}^n \mid v_i^TP^{-1}v_i \leq 1, \quad i = 1,\cdots, k\}，
$$
即定义了一群包含点$v_0, \cdots, v_k$的椭圆集合S。如下图所示，此时S没有最小元，因为当有椭圆$\varepsilon_1$包含这些点，我们也能找到另外一个椭圆
$\varepsilon_3$包含这些点，但是这两个椭圆之间无法比较（因为我们只定义了$\varepsilon_A \subseteq \varepsilon_B$这层关系，
除非再限定了其他可以比较的条件：eg，椭圆面积等，否则无法比较大小）；但是却可以找到极小元$\varepsilon_2$，S中没有任何其他椭圆可以被它包含了。

![图2-10](/img/posts/convex optimization/chapter two/chapter_two_10.PNG)


**注**

&ensp; 上面广义不等式$\preceq_K$、$\prec_K$和最小元、极小元的概念其实很自然联想到在R上的不等式，例如:$\leq$、$<$和最小值、极大值，
但是它们之间是有共同点和差异点的，最大的差异就是：

+ 在R上的不等式为线性序，即任何两点都是可以比较的，但是广义不等式不一定都可比较，例如上图中的椭圆$\varepsilon_1$和$\varepsilon_3$。

+ 在R上极大值和极小值相对之间都是可以比较的，代表的是局部的最大和最小的意义。

### **Part 5 : 2.10. 分离和支持超平面(separating and supporting hyperplane)**

**分离超平面**

假设C与D为非空的不相邻的凸集，$C \cap D = \phi $，$\exists a \neq 0, b$，对于所有的$x \in C$，有$a^Tx \leq b$；
对于所有$x \in D$，有$a^Tx \geq b$，则超平面$$\{x \mid a^Tx = b \}$$为分离超平面。

proof: 假设在欧几里得距离度量下，存在$c \in C, d in D$，使$\parallel C-d \parallel_2 = dist(C,D)$取得最小，
不妨定义$a = d -c, \quad b = \frac{\parallel d \parallel_2^2 - \parallel c \parallel_2^2}{2}$，则
$f(x) = a^Tx - b = (d-c)^T(x-1/2(d+c))$为分离超平面。下图是一个例子：

![图2-11](/img/posts/convex optimization/chapter two/chapter_two_11.PNG)

>Strict seperation : $\forall x \in C，a^Tx < b ；\forall x \in D， a^Tx > b$

**支持超平面**

假设$C \subseteq R^n$， $x_0$为C边界上的点，即：$x_0 \in bd \ C = cl \ C \setminus int \ C$，
若$a \neq 0$满足对任意$x \in C，a^Tx \leq a^Tx_0$，则超平面$$\{x \mid a^Tx = a^Tx_0 \}$$称为C在$x_0$点处的支持超平面。
若从集合角度看，$$\{ x \mid a^Tx = a^Tx_0 \}$$是点C处的正切平面。下图是一个例子：

![图2-12](/img/posts/convex optimization/chapter two/chapter_two_12.PNG)

**性质**

1. 对于任何非空的凸集C，且任何的$x_0 \in bd$，都存在一个过$x_0$的支持超平面  

2. 一个内部非空封闭的集合，若在边界上的每一个点均存在支持超平面，则该集合为凸  

&ensp; 这两个性质还是比较容易得到的，给个比较通俗**但不是很严谨的证明**：假设C为有边界的凸集，由凸集的性质知：$$\theta x_1 + (1-\theta)x_2，\theta \geq 0$$，则对于边界上的点$x_1$，$x_2$沿边界无限逼近$x_1$的时候，此时由$x_1，x_2$组成的直线可以代替过它们的边界线，则此时当$x_2 \rightarrow x_1$时，无论$x_1$点处是不存在切线(eg:$y \geq \mid x \mid，x_1 = 0$)，还是有切线$\iota$，都能找到一个超平面使所有点要么在这条直线上方，要么在下方。可见图：

![图2-13](/img/posts/convex optimization/chapter two/chapter_two_13.PNG)


### **Part 6 : 2.11. 对偶堆与广义不等式（dual cones and generalized inequalities)**

**对偶锥(Dual cones)**

&ensp; 令K为堆，则集合：$$ K^* = \{ y \mid \forall x \in K, \quad x^Ty \geq 0 \}$$为K的对偶堆，且有性质：
无论K是否为凸，$ K^* $总是为凸锥。从几何角度来看：$y \in K^* $, 当且仅当$-y$为K在原点的支持超平面的法线：

![图2-14](/img/posts/convex optimization/chapter two/chapter_two_14.PNG)

&ensp;如上图，(a)中$ y \in K^* $，因为其代表的半空间包含$K$，即$-y$为$K$在原点的支持超平面的法线，而(b)中 $ z \notin K^* $，因为其代表的半空间有一部分不包含$K$，即不是$K$在原点的支持超平面的法线。

**自对偶(Self-daul)**

&ensp; 满足$$ \forall x, x^Ty \geq 0  \Leftrightarrow y \succeq 0 $$ 称为自对偶，例如锥$R_+^n$、半正定锥$S_+^n$均为self-dual

**对偶锥满足的一些性质:**

1. $K^* $是封闭且为凸

2. $K_1 \subseteq K_2$，则$K_2^* \subset K_1^* $：这一点从对偶有点类似补集的概念理解

3. 若$K$有非空内点，则$K^* $ is pointed.

4. 若closure of K is pointed, 则$K^* $有非空内点

5. $K^{** }$ is the closure of the conve hull of K，即$K^{** } = K$

**对偶广义不等式**

+ $ x \preceq_k y$，当且仅当：$\forall \lambda \succeq_{k^* } 0,\quad \lambda^T x \leq \lambda^Ty $

+ $ x \prec_k y$当且仅当：$\forall \lambda \succeq_{k^* } 0, \lambda \neq 0, \quad \lambda^Tx < \lambda^Ty $

其中，$\preceq_{k^* } $为广义不等式$\preceq_k$的对偶形式，由$K^* = K$，则$\lambda \preceq_{k^* } \mu$，当且仅当$\forall x \succeq_k 0, \quad \lambda^Tx \leq \mu^Tx$

> 这章本人花了大量时间整理，以便后面章节能理顺一些基本公式的推导过程，文中难免有局限于当时的历史因素，可能存在不同程度的错误，若有童鞋发现，欢迎指出。
> 文中所有的图和公式都是本人用Word和Latex公式制作，若有童鞋要用到欢迎到[zbabby](https://github.com/tianbaochou/tianbaochou.github.io/tree/master/img/posts)的convex optimization相应章节录中下载。希望大家使用的
时候可以注明出处，谢谢!
