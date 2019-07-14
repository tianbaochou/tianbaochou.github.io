---
layout: post
title: 逻辑回归分类器(Logistic Regression)
date: 2019-07-13 20:10
description: 本专题将记录逻辑回归分类器
tag: deep learning
---

LR分类器几乎是机器学习面试必考的点，究其原因，我想是因为它的梯度计算公式不是特别好算吧
逻辑回归虽然称为回归，**但是却是用来分类的**，其不同于线性回归，它是将线性回归的输出值作为输入，然后判断其类别，常用在二分类上。大家可能会想到，那么如果是多分类呢？ 这里最简单的方法就是用我们上一节讲到的，用Hinge Loss或者Softmax Loss。下面就简单介绍一下LR，并在博客上手推一遍代码。


## **1. Sigmoid函数**

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

上面的$z$是对输入进行线性变换后的输出值，也就是线性回归的输出值！则Sigmoid的导数为

$$
\sigma(z)^{'} = \sigma(z)(1-\sigma(z)) 
$$

这里就不推了，因为这个结果很优美，所以我就偷懒，直接把它记到脑子里拉。首次建议大家推一下。
那么Sigmoid函数和Logistic Regression之间是什么关系呢？我们看看下面这张图就明白了

![img1](img/posts/deep learnig/LR.svg)

相当于说，sigmoid为逻辑回归的激活函数！通过sigmoid函数的作用，我们可以将输出值$f$限定到$[0,1]$区间内，以二分类为例，如果输出$f > 0.5$，我们可以认为$A$类，反之，为$B$类

## **LR具体步骤**

+ 寻找预测函数$h$
+ 构造损失函数$L$
+ 迭代更新参数


### 寻找预测函数$h$

以二分类为例，如果$h(x)$代表取$y=1$的概率，则$1-h(x)$代表取$y=0$的概率，这样$h(x)$可以看成类1的后验概率分布,故:

$$
\begin{aligned}

p(y=1 \mid x; \theta) = h(x)  \\
p(y=0 \mid x; \theta) = 1 - h(x)

\end{aligned}
$$

### 构造损失函数$L$

由于我们的预测函数是一个概率值，只有$h(x)$和$1-h(x)$输出，我们希望的是$h(x)$概率趋近与$y=1$，即一个0-1分布，也称伯努利分布。因此，可以使用交叉熵损失:

$$
L = -\sum_{i=1}^{N}y_{i}log(\sigma(f_{i})) + (1-y_{i})log(\sigma(1-f_{i}))
$$

现在我们需要手推一下其梯度值

$$
\begin{aligned}
\Delta_W L = \sum_{i=1}^{N} [ y_i \frac{\sigma(f_i)^2(1-\sigma(f_i))*x_i}{\sigma(f_i)} + (1-y_{i})\frac{\sigma(1-f_{i})^2(1-\sigma(1-f_{i}))*(-x_i)}{(\sigma(1-f_i))}
 ] \\

= \sum_{i=1}^{N}[y_i(\sigma(f_i)-\sigma(f_i)^2)*x_i + (1-y_i)(\sigma(1-f_i)-\sigma(1-f_i)^2)*(-x_i)] \\
= -\sum_{i=1}^{N}((y_{i} - \sigma(f_i))x_i) = -\sum_{i=1}^{N}error_{i}x_i

\end{aligned}
$$

有了上面的推导，我们可以有如下LR的代码:

```python

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def naive_gradient_desc(X, Y, eta, iter):
    """
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = 1 means A

    Returns:
    - loss
    - gradient with respect to weights W; an array of same shape as W
    """
    N, D = X.shape
    W = np.ones((D, 1)) #(D,1) 初始化为1
    for i in range(iter):
        f = sigmoid(X.dot(W)) # N x 1
        error = f - y         # N x 1
        W = W - eta*X.T*error # D x 1
    return W
```

由于逻辑回归使用交叉熵损失函数，故其同Softmax线性分类器一样可以看成极大似然估计，且
