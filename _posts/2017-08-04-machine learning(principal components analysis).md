---
layout: post
title: principal components analysis
date: 2017-08-04 21:21
description: 本篇专题将会详细讲解PCA的基本原理以及涉及的代码tricks
tag: machine learning
---

## **1. 前言**

&ensp; PCA是本人比较早接触的一个tool，做数模时在统计部分经常用。惭愧的是当时没有总结PCA数学背景（统计背景），只是知道PCA可以对一组数据
筛选出主要的成分（特征），并可以从中看出对应维度的重要性。这两天找了许多的资料已经能够比较深刻的理解PCA这个tool。下面我们将先从一个例子[1]引入，然后引出PCA算法，最后用python编写代码，并作图对比PCA降维后的数据和原数据在图像上的关系。

## **2. PCA**

###  **2.1 An Example**

 假设有10组数据$X$，其特征维度为2:

 ![图1](/img/posts/machine learning/pca/img1.JPG)

 我们为了使数据看起来更加'对称'，首先将数据$X$减去平均值$\bar{X}$得到：

 ![图2](/img/posts/machine learning/pca/img2.JPG)

这个做法相当于将原先的坐标轴移动到下图的mean位置。 那么还有没有更好的一个坐标轴能够**使数据投影在该坐标轴上的方差总和最大**。
也许你会问为什么要使数据投影在该坐标轴上的方差总和最大，这是因为当方差总和最大时，一般数据点投影后重合率较小，即在该坐标轴上
数据可以被很好的分类。这个坐标轴就是如下图的粗虚线的主轴。

![图3](/img/posts/machine learning/pca/img3.JPG)

这个主轴就是PCA中的第一个轴，在本例中，如果我们将2维的数据降为1维，那么降维后的数据值就是在主轴上的投影值。
现在我们的问题是如何找到这个主轴，下面我们将引出PCA的原理。

## **2.2 PCA**


假设我们有数据集$\{x_{n}，n = 1,2,...,N\}.$，数据的维度为$D$。现要将数据投影到一维空间，并定义此空间的方向由D维向量$u_1$定义。为了后面的计算方便，我们令$u_{1}$为单位向量。则每一个数据点$x_n$投影到$u_1$的值为$u_{1}^{T}x_n$，投影后这群点的中心位置为$u_{1}^{T}\bar{x}$。投影后点的方差定义为：

$$
\frac{1}{N-1} \sum \limits_{n=1}^{N}{(u_{1}^{T}x_{n} - u_{1}^{T}\bar{x})}^{2} = u_{1}^{T}Su_{1} \tag{1}
$$

其中S为协方差矩阵:

$$

S = \frac{1}{N-1}\sum \limits_{n=1}^{N}(x_{n} - \bar{x})(x_{n} - \bar{x})^T \tag{2}

$$

> 注意这里的方差，我们不取： $ S = \frac{1}{N}\sum \limits_{n=1}^{N}(x_{n} - \bar{x})(x_{n} - \bar{x})^T $
具体原因是由于当我们研究一个整体时，通常会选择样本来估计出总体的方差$\sigma^{2}$，根据估计量的评选标准之一 **无偏性**，且当样本方差为S除以$N-1$时，$E(S) = \sigma^{2}$，即满足无偏性。相反，当样本方差为除以N时，不满足无偏性[2]


则我们转为如下问题:

$$

max  \quad u_{1}^{T}Su_{1}  \\

subject \quad to \\

u_{1}^{T}u_{1} = 1 \tag{3}

$$

转为拉格朗日问题：

$$

L(u_{1},\lambda) = u_{1}^{T}Su_{1} + \lambda_{1}(u_{1}^{T}u_{1} -1) \tag{3}

$$

对$u_{1}$求偏导得到：

$$
Su_{1} = \lambda_{1} u_{1} \tag{4}
$$

从(4)可以看出$u_{1}$为协方差矩阵S对应的特征值为$\lambda_{1}$的特征向量。
再将(4)两边同乘$u_{1}^T$，得到 $u_{1}^{T}Su_{1} = \lambda_{1}$，左边即是我们要最大化的方差。
故当$\lambda_{1}$取最大时，该方差最大。总结来讲，我们要取得到的$u_{1}$即为 **协方差矩阵$S$的特征值最大的单位特征向量!**
称$u_{1}$为 **first principal component**。推广一下，当我们要将数据投影到2维空间时，我们可以选取和$u_{1}$垂直的第二大特征值对应的单位特征向量。推广到M维，则选取和$u_{1}, u_{2}, ..., u_{M-1}$向量均垂直的第M大的特征值对应的单位特征向量。

最后假设我们的数据集$\{x_{n}，n = 1,2,...,N\}.$放在矩阵$X$内，每一行为一个数据，每一列为一种特征。不难得出中心化的数据集$(X-\bar{X})$投影在特征向量空间上的坐标为：

$$

X_{proj} = (X-\bar{X})U \\
U = [u_{1}, u_{2}, ..., u_{D}] 为单位特征向量 \tag{5}

$$

综上，我们可以看出PCA的作用在几何上为：**旋转坐标轴，使数据取得高方差的轴最先选取**

## **3. PCA code**

从上面的数学原理中可以看出PCA中涉及的两大点就是求协方差矩阵S和S的特征值与特征向量（python的线性代数包中已经单位化）的求解。
故代码很简单，如下[4]：

~~~

def pca(dataMat, topNfeat = 999999):
    meanVals = mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0) # equal to (1/[n-1])*(meanRemoved.T*meanRemoved)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    print eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    # get the data old data back to show which data we choose[if we reduce some dim, then the retrieved data has lost] some information]
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


~~~

上面代码 ``` reconMat = (lowDDataMat * redEigVects.T) + meanVals ``` 表示将降维的数据重新映射回原先的坐标系。由[3]中知:

$$

X_{back} = (X - \bar{X})UU^T + \bar{X}  \\
U = [u_{1}, u_{2}, ..., u_{D}] 为单位特征向量 \tag{6}
$$

这个推导并不难，这里就不写出来了。我们从中可以看出，当PCA没有选择降维时，$U^{T}U = 1$故$X_{back} = X$，即
重新映射回原先的坐标系后和原先的一致，没有丢失信息!
为了方便比较$X_{back}$与$X$。我们降一个二维的数据集，降到一维，并重新映射回原先的坐标系，得到如下：

![图四](/img/posts/machine learning/pca/img4.png)

上图中红色的部分为重新投影回的数据，可以看出，该数据的方向与本专题引入的例子first principal component类似。

[1] Lindsay I Smith. A tutorial on Principal components analysis.

[2] 概率论与数理统计(第四版). p159.

[3] Christopher M. Bishop. CHAPTER 12 Principal Component AnalysisPattern. Recognition and Machine Learning 560-564.

[4]  Peter Harrington. CHAPTER 13 Using principal component analysis to simplify data. Machine learning in action:273-274.
