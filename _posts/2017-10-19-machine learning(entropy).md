---
layout: post
title: entropy
date: 2017-10-19 20:00  
description: 本专题力图将机器学习中用到的熵的知识串起来，以便理清脉络
tag: machine learning
---

# **1. Entropy**

## **1.1. Entropy定义**

在**信息论与概率统计**中，Entropy(熵)表示随机变量不确定性的一种度量，设$X$是有限离散的随机变量，其概率分布为：

$$

P(X = x_i) = p_i, \quad i = 1,2,\cdots,n

$$

则随机变量$X$的熵的定义为：

$$
H(X) = -\sum \limits_{i=1}^{n}p_ilog(p_i) \tag{1}
$$

&ensp;例如当$X$仅能取0和1，且$p_1 = p_2 = \frac{1}{2}$时，$H(x) = 1$，即表示只用一个bit位就能
表示$X$。(1)中的$log$以2为底或以e为底时，熵的单位分别称为bit或者nat(纳特)。我们通常取2为底，后面都将以2为底!

&ensp;由定义知，熵越大，随机变量的不确定性就越大。

## **1.2. Conditional Entropy定义**

&ensp;设有随机变量$(X,Y)$，其联合概率分布为  

$$

P(X = x_i, Y = y_i) = p_{ij}, \quad i = 1,2,\cdots,n; \quad j=1,2,\cdots, m

$$

条件熵(conditional entropy) $H(Y\|X)$ 表示在 **已知随机变量 $X$** 的条件下随机变量 $Y$ 的不确定性，其定义为：

$$

H(Y|X) = \sum \limits_{i=1}^{n} p_{i}H(Y|X=x_i) \tag{2}

$$

&ensp;这里，$p_i = P(X=x_i)$， 可见其表示为在$X$给定的条件下$Y$的条件概率分布的熵**对$X$的数学期望**。
当熵和条件熵中的概率由数据估计（例如是极大似然估计）得到时，所对应的熵和条件熵分别称为**经验熵**和**经验条件熵**

## **1.3 Information Gain定义**

&ensp;信息增益(information gain)表示得知特征$X$的信息而使得类$Y$的信息的不确定性减少的程度：

$$

g(D,A) = H(D) - H(D|A)  \tag{3}

$$

&ensp;以决策树为例，当给定数据集D和特征A时，若A给定的情况下对信息的分类的不确定性$H(D|A)$**比较小**，
则易知$g(D,A)$相对比较大，说明该特征对整个数据的分类能力比较强!!

## **1.4 Information Gain Rdtio**

&ensp; 上面说过信息增益比较大的特征$A$，分类能力相对较强，因此决策树会偏向于选择该特征，这将造成
偏向于**选择取值较多的特征**问题。使用信息增益比(information gain ratio)可以对这一问题进行校正,
其定义为信息增益$g(D,A)$与训练数据集D关于A的值的熵$H_A(D)$之比：

$$

g_R(D,A) = \frac{g(D,A)}{H_A(D)} \tag{4}

$$

&ensp;其中，$H_A(D) = -\sum \limits_{i=1}^{n}\frac{\vert D_i \vert} {\vert D \vert}log_2\frac{\vert D_i \vert}{\vert D \vert},n$是特征$A$取值
的个数。

## **1.5 交叉熵和相对熵**

&ensp;现在我们假设有两个概率分布P和Q,其中P为实际分布，Q为我们拟合或者预测的分布，那么如何按照实际分布P去比较和Q的
相似性（或者差异性）？为此我们定义了**相对熵**：

$$

D(P||Q) = H(P,Q) - H(P) = \sum \limits_{x} P(x)log\frac{P(x)}{Q(x)} \tag{5}

$$

它又被称为KL散度(Kullback-Leibler divergence,KLD)。其中$$ H(P,Q) = \sum P(x)log\frac{1}{Q(x)} $$,
H(P,Q)称为**交叉熵(cross entropy)** 。根据[Gibbs'inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality)，D(P||Q)
恒大于等于0。下面举个例子：

&ensp; 假设在四个单词的数据集D={'You', 'I', 'He', 'She'}中，实际分布$$P=\{ \frac{1}{2}, \frac{1}{2}, 0, 0 \}$$，计算
$H(P) = -\sum \limits_{i=1}^{4}P(x_i)log\frac{1}{P(x_i)} = \frac{1}{2} + \frac{1}{2} = 1$，即仅需要1 bit即可识别'You'和'I'。
现若我们预测D的分布为$$Q=\{\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\}$$，则
$H(P,Q) = -\sum \limits_{i=1}^{4} P(x_i)log\frac{1}{Q(x_i)} = 2$，即需要2 bits才可识别'You'和'I'。
此时的KL散度为1。

KL散度不是对称的，因为P对于Q的KL散度并不等于Q对于P的散度。所以实际上Kullback-Leibler定义了另外一个形式的散度：

$$
D_{KL}(P||Q) + D_{KL}(Q||P)
$$

这样就解决了不对称问题了，另外还有$\lambda$散度[Kullback–Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$

D_{\lambda}(P||Q) = \lambda D_{KL}(P||\lambda P+(1-\lambda)Q)+(1-\lambda)D_{KL}(Q||\lambda P+(1-\lambda)Q) \tag{6}

$$

当$\lambda = 0.5$，就得到了**Jensen-Shannon** divergence：

$$

D_{JS} = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)

$$

其中，M为两个分布的平均$M=\frac{1}{2}(P+Q)$。

参考资料：

1. 李航，统计学习方法
2. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
3. https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence#cite_note-2
3. https://www.zhihu.com/question/41252833
