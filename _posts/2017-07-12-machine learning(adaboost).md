---
layout: post
title: adaboost
date:  2017-07-12 16:00
description: adaboost with decision stump
tag: machine learning
---

**概要**

提升方法(boosting method)是一种在分类问题中通过不断改变样本权重和学习多个分类器，并根据每个分类器的分类效果，更新分类器权重，最终通过加权分类器分类结果以得到最终分类结果。本文主要讲典型提升算法：AdaBoost算法。

## **1. AdaBoost算法**

本来想直接看Freund Y, Schapire的原论文，但是瞄了一眼Paper，发现里面关于Loss的分析太繁琐。因此后面将主要介绍论文脉络，以后有时间再啃啃。AdaBoost算法属于Ensemble method，而Ensemble method的基本思想就是利用若干个Weak Learner合作解决问题。故AdaBoost算法其实借鉴了很多前人的工作，将很多前辈的巧妙想法融合。
在正式讲AdaBoost算法之前，我们来看看一些不可或缺的前人工作：

1. 1988年M.Kearns and L.G. Valiant提出一个问题：一个Weak learning算法是否可以提升（boosting)成一个任意精度的strong learning

2. R.Schapire的一篇论文中(MLJ90)给出了上述问题肯定的答案，并给出了证明。（也是第一个Boosting的产生）

1997年Freund和前面那个证明的家伙Schapire给出了Ababoost algorithm[1]。其算法的前生是$$Hedge(\beta)$$，该算法虽然并不多人知道，但是却是Adaboost algorithm的核心框架。

下面介绍Adaboost algorithm的步骤：


假设训练数据集$T = {(x_1, y_1), (x_2,y_2), ..., (x_N, y_N)}$，其中$x_i \in X \subseteq R^n$， $$y_i \in Y =\{-1, 1\}$$ （**原论文中$y_i$ = {0, 1},跟这里不同，这里这样改更简单**），
迭代总次数为T，弱分类器为$G(x)$。

+ 1.初始化训练数据的权值分布：

$$
D_1 = (w_{11}, w_{12}, ..., w_{1N})，w_{1i} = \frac{1}{N}，i = 1, 2, ..., N \tag{1}
$$

+ 2.由1到T迭代，假设当前迭代次数为$t$：

(A) 用有权值分布$D_t$的训练数据集学习，得到基本分类器：

$$
G_{t} : X \Rightarrow \{-1, 1\} \tag{2}
$$

(B) 计算$G_{t}(x)$在训练数据集上的分类误差率：

  $$
  \epsilon_{t} = P(G_{t}(x_i) \neq y_i) = \sum\limits_{i=1}^{N}w_{ti}I(G_{t}(x_i) \neq y_{i}) \tag{3}
  $$

(C) 计算$G_{t}(x)$的系数：

$$
\alpha_{t} = \frac{1}{2}\log{\frac{1-\epsilon_{t}}{\epsilon_{t}}} \tag{4}
$$

由上面这个系数计算公式易知，当$\epsilon_{t} = \frac{1}{2}$时，$\alpha_{t} = 0$。因为此时$G_{t}(x)$的分类结果**相当于我们随机猜测的结果**，错误率为$\frac{1}{2}$，故**$G_{t}(x)$无效!!**
同时当$\epsilon_{t} < \frac{1}{2}$时，$\epsilon_{t}$越小，$\alpha_{t}$越大，且大于0；$\epsilon_{t} > \frac{1}{2}$的情况并不会出现，因为我们总能选择其相反的划分，使得$\epsilon_{t} < \frac{1}{2}$。

(D) 更新训练数据集的权值分布：

$$
\begin{eqnarray*}
&& D_{t+1} = (w_{(t+1,1)}, ..., w_{(t+1,N)})   \\
&& w_{(t+1,i)} = \frac{w_{(t,i)}}{Z}e^{-\epsilon_{t}(y_{i}G_{t}(x_i))}，\quad i=1，2，...，N \\
&& Z = \sum \limits_{i=1}^{N}{w_{(t+1,i)}}
\end{eqnarray*}\tag{5}
$$

其中$Z$的作用是规范化$w_{(t+1,i)}$，使$D_{t+1}$成为一个概率分布(即更新后的w之和为1)。可见当$G_{t}(x_i) = y_i$时其$w_{(t+1,i)}$比较小，反之，比较大。即下一次
训练样本的权重的分配情况是：**本次分类被误分的样本权重增大；正确分类的样本权重减小。**

+ 3.将弱分类器线性组合：

$$
f(x) = \sum \limits_{t=1}^{T}\alpha_{t}G_{t}(x) \tag{6}
$$

得到最终的分类器:

$$

F(x) = sign(f(x)) = sign(\sum \limits_{t=1}^{T}\alpha_{t}G_{t}(x))\tag{7}

$$

由上面的步骤可以看出:Adaboost在训练的过程中是不改变训练数据，仅仅改变训练数据的权值分布和对应的弱分类器在构成线性组合中的系数。
当迭代此时达到预设的T或者本次线性组合的分类器$F_{t}(x)$分类错误率为0时结束。

### **2.1. 弱分类器（Weak Learner）**

上面，我们用到的弱分类器，一般来说越简单越好。这是因为，根据理论[1]弱分类器越简单越可以避免最终的F(x)过拟合，导致在测试集上错误率大。
**算法可以选择不同的Weak Learner，也可以每次都用相同的**，一般常见的Adaboost选择的Weak Learner 是 **decision stumps**。
其具体分类步骤的伪代码如下：

~~~

minError = +inf
set numSteps

For every feature in dataset:
    For every step:
        For each inequality:
            在带权重的训练数据集中建立decision stump并计算error
            If error < minError:
                Set this stump as the best stump
return the best stump

~~~

可以从伪代码中看出，decision stump仅选择最好的一种特征来分离数据，因此称为stump。下面我们给出一种stump的代码，以便于理解：

~~~

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):

    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMin - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j)*stepSize) # 范围取 [rangeMin - stepSize, rangeMax + stepSize]
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0   # 当预测值与真实值不等时，errArr相应位置为1
                weightedError = D.t*errArr     
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy() # python 中list为引用,故需深度拷贝
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst
~~~

### **2.2. Adaboost alogorithm的一个实例**

为了加深对Adaboost算法计算步骤的理解，我们引用[2]中的例子来说明问题：


假设有如下表的训练数据，每个数据仅有一个特征。


| 序号  | 1  | 2 |  3  | 4 |  5 |  6 |  7 |  8 |  9  | 10  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| x      | 0  |  1 |  2  | 3  | 4  | 5 |  6  | 7  | 8  | 9     |
| y      | 1   | 1   |1  | -1 | -1  | -1  |1  | 1  | 1 |  -1   |

则由Adaboost algorithm算法的步骤如下：

初始化权值分布:

$$

D_{1} = (w_{(1,1)}, w_{(1,2)}, ... w_{(1,10)})  \\
w_{(1,i)} = 0.1，i = 1,2,...,10

$$

+ **1.当t = 1时：**

(a) 在权值分布为$D_1$的训练数据上，阈值threshVal取2.5，threshIneq = 'gt'时分类误差最小，故弱分类器为:

$$
G_1{x} = \left\{
    \begin{align*}
    1, \quad x <= 2.5 \\
    -1, \quad x > 2.5
    \end{align*}
    \right.
$$

(b) $G_{1}(x)$在训练数据集上的误差率为$\epsilon_1 = P(G_{1}(x_i) \neq y_i) = 0.3$

(c) 计算$G_{1}(x)$的系数： $\alpha_{1} = \frac{1}{2}\log\frac{1-\epsilon_{1}}{\epsilon_{1}} = 0.4236$

(d) 更新训练数据的权值分布：

$$
\begin{eqnarray*}
&& D_{2} = (w_{(2,1)}, ..., w_{(2,i)}, ..., w_{(2,10)})  \\
&& w_{(2,i)} = \frac{w_{(1,i)}}{Z_{1}}e^{-\alpha_{1}y_{i}G_{1}(x_i)}, \quad i = 1, 2, ..., 10 \\
&& D_{2} = (0.07143, 0.07143, 0.07143, 0.07143, 0.07143,0.07143,0.16667,0.16667,0.16667,0.07143) \\
&& f_{1}(x) = 0.4236G_{1}(x)
\end{eqnarray*}
$$

分类器$sign(f_{1}(x))$在训练数据集上有3个误分类点。

+ **2.当t = 2时：**

(a) 在权值分布为$D_{2}$的训练数据集上，阈值threshVal取8.5，threshIneq = 'gt'时分类误差最小，弱分类器为：

$$
G_2{x} = \left\{
    \begin{align*}
    1, \quad x <= 8.5 \\
    -1, \quad x > 8.5
    \end{align*}
    \right.
$$

(b) G_{2}(x)在训练数据集上的误差率为$\epsilon_2 = P(G_{2}(x_i) \neq y_i) = 0.2143$

(c) 计算$G_{2}(x)$的系数： $\alpha_{2} = \frac{1}{2}\log\frac{1-\epsilon_{2}}{\epsilon_{2}} = 0.6496$

(d) 更新训练数据的权值分布：

$$
\begin{eqnarray*}
&& D_{3} = (w_{(3,1)}, ..., w_{(3,i)}, ..., w_{(3,10)})  \\
&& w_{(3,i)} = \frac{w_{(2,i)}}{Z_{2}}e^{-\alpha_{2}y_{i}G_{2}(x_i)}, \quad i = 1, 2, ..., 10 \\
&& D_{3} = (0.0455, 0.0455, 0.0455, 0.1667, 0.1667, 0.1667,0.1060,0.1060,0.1060,0.1060,0.0455) \\
&& f_{2}(x) =  0.4236G_{1}(x) + 0.6496G_{2}(x)
\end{eqnarray*}
$$

分类器$sign(f_{2}(x))$在训练数据集上有3个误分类点。

+ **3.当t = 3时：**

(a) 在权值分布为$D_{3}$的训练数据集上，阈值threshVal取5.5，threshIneq = 'lt'时分类误差最小，弱分类器为：

$$
G_3{x} = \left\{
    \begin{align*}
    1, \quad x <= 8.5 \\
    -1, \quad x > 8.5
    \end{align*}
    \right.
$$

(b) G_{3}(x)在训练数据集上的误差率为$\epsilon_3 = P(G_{3}(x_i) \neq y_i) = 0.1820$

(c) 计算$G_{3}(x)$的系数： $\alpha_{3} = \frac{1}{2}\log\frac{1-\epsilon_{3}}{\epsilon_{3}} = 0.7514$

(d) 更新训练数据的权值分布：

$$
\begin{eqnarray*}
&& D_{4} = (w_{(4,1)}, ..., w_{(4,i)}, ..., w_{(4,10)})  \\
&& w_{(4,i)} = \frac{w_{(3,i)}}{Z_{3}}e^{-\alpha_{3}y_{i}G_{3}(x_i)}, \quad i = 1, 2, ..., 10 \\
&& D_{4} = (0.125, 0.125, 0.125, 0.102, 0.102, 0.102,0.065, 0.065, 0.064, 0.125) \\
&& f_{3}(x) =  0.4236G_{1}(x) + 0.6496G_{2}(x) + 0.7514G_{3}(x)
\end{eqnarray*}
$$

分类器$sign(f_{3}(x))$在训练数据集上有0个误分类点，迭代结束。
于是最终的分类器为:

$$

F(x) = sign(f_{3}(x)) = sign(0.4236G_{1}(x) + 0.6496G_{2}(x) + 0.7514G_{3}(x))

$$

其更新步骤示意如下图：

![图一](/img/posts/machine learning/adaboost/img1.jpg)


## **2. AdaBoost算法核心代码**

~~~


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []                # 用于存储每次更新的弱分类器的系数等重要数据
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)           # 初始化训练样本权值
    aggClassEst = mat(zeros((m,1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D,exp(expon))
        D = D /D.sum()                # 规范化
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print('total error: {0}'.format(errorRate))
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        # dataToClass可以是多个测试数据
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

~~~

该段代码简单易懂，没有SVM那样在计算时要用到很多公式，但是adaBoost的损失函数的分析并不简单。
作者在Paper中用了大量篇幅证明该算法可以**在每次更新参数后，以指数级别的下降速度接近于最优值**。这也是我们在实际应用中并不会担心算法会不收敛的原因，建议有能力的读者可以跟着作者去推导损失函数部分内容。

[1] Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting.

[2] 李航. 统计学习方法-第八章 提升方法.

[3]  Peter Harrington. CHAPTER 7 Improving classification with the adaBoost meta-alogirithm. Machine learning in action:130-148.
