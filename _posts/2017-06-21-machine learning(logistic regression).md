---
layout: post
title: logistic regression
date: 2017-06-21 15:03
description: logistic regression essential formula
tag: machine learning
---

# **1. Logistic Regression**

基本的Logistic Regression 实际上简单易懂，即对于数据X和Y，寻找一个最佳的hyperplane，可以最好的区分两个**数据**。
这里所谓的**最好**，即代价函数(cost function)最小。以二分类0-1问题为例，在训练数据中每个数据的类别为0或者1，则我们需要
拟合一个函数可以对输入数据X，预测其属于0或者1，这个函数称为Heaviside step function或者Step function。这个函数并不好处理，
因此人们想到用其他图像近似但好处理的函数来替代它，一般有Logistic function(or sigmoid function)和tanh。我们这里
就介绍logistic function。



## **1.1. Logistic Function(sigmoid function:)**

$$

h_{\theta}(x^{(i)}) = \frac{1}{1 + e^{-\theta^T x}}

$$

其图像如下:

![图1](/images/posts/machine learning/machine learning foundation/logistic regression/image1.jpg){:with=”500px” height=”422px”}

## **1.2. Cost Function**

$$
C(h_{\theta}(x), y) = \left\{
\begin{gather*}

-log{(h_{\theta}(x))}  \qquad if \qquad y = 1 \\
-log(1-h_{\theta}(x))  \qquad if  \qquad y = 0
\end{gather*}
\right.
$$

 当**y=1**时,其函数图像类似如下:

![图2](/images/posts/machine learning/machine learning foundation/logistic regression/image2.jpg)

从上图可以看出当$h_{\theta}(x)$接近1时，代价接近0, 当$h_{\theta}(x)$接近0时，
其代价无穷大。


当**y=0**时,其函数图像类似如下:

![图3](/images/posts/machine learning/machine learning foundation/logistic regression/image3.jpg)

从上图可以看出当$h_{\theta}(x)$接近0时，代价接近0, 当$h_{\theta}(x)$接近1时，
其代价无穷大。

将上面分段的代价函数统一为： $  y^{(i)}log{(h_{\theta}(x^{(i)}))} + (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))  $，则对于
m个训练数据的平均代价为：

$$
\begin{align*}
J(\theta)  & =   \frac{1}{m} \sum_{i=1}^{m}{C(h_\theta(x^{(i)}), y^{(i)})} \\
  & =  -\frac{1}{m} \sum_{i=1}^{m}{y^{(i)}}log{(h_{\theta}(x^{(i)}))} + (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))
\end{align*}
\tag{1-4}
$$

## **1.3. Gradient Of Cost Function About $\theta$**

$$

\begin{align*}
\nabla{J_{\theta}(\theta)} & = -\frac{1}{m}[\sum_{i=1}^{m}{y^{(i)}\frac{h_{\theta}^{'}}{h_{\theta}(x^{(i)})} +
(1-y^{(i)})\frac{-h_{\theta}^{'}(x^{(i)})}{1-h_{\theta}(x^{(i)})}}] \\

&= \frac{1}{m}\sum_{i=1}^{m}{\frac{[h_{\theta}(x^{(i)}) - y^{(i)}]h_{\theta}^{'}(x^{(i)})}{h_{\theta}(x^{(i)})(1-h_{\theta}(x^{(i)}))}} \\
&= \frac{1}{m}\sum_{i=1}^{m}{[h_{\theta}(x^{(i)}) - y^{(i)}]x^{(i)}}
\end{align*}

$$

因为$\frac{1}{m}$为正数，不影响梯度方向，故忽略。最终$ \nabla_{\theta}{f(\theta)} $ 为: $$ \sum_{i=1}^{m}{[h_{\theta}(x^{(i)}) - y^{(i)}]x^{(i)}} $$

# **2. Gradient Descent**

在求函数最小值的时候(如求代价函数最小)，更新函数参数w(即上面的$\theta$)的依据是沿着函数梯度的负方向可以以最快的速度到达最小值:

$$ w = w - \alpha \nabla_{w}{f(w)} $$

若要求函数最大值，则易知仅需变为:

$$ w = w + \alpha \nabla_{w}{f(w)} $$

其中$\alpha$ 为 step size 或者 learning rate

若在python中借助numpy的矩阵运算可以很简洁的写出梯度下降的核心代码:

~~~
from numpy import *

def sigmoid(intX):
    return 1 / (1 + exp(-intX))

def gradDescent(datasIn, classesIn):
    matDatas = mat(dataIn)     # m*n
    matClasses = mat(classesIn).transpose() # 转为m*1
    m, n = shape(matDatas)
    w = ones((n,1)) # w = (w_1,w_2,...w_n)为权重即上面的theta
    maxCycles = 500
    alpha = 0.001

    for i in range(maxCycles):
        h = sigmoid(matDatas*w)
        error = classesIn - h  # 即上面的y-h
        w += alpha*matDatas.transpose()*error # 更新梯度
    return w
~~~

其中dataIn为$ m \times n$即m个n维训练数据， classIn为$m \times 1$即相应的m个类别（0或者1)。


> 本段代码引自 <machine learning in action> 关于logistic regression的代码段，文中称为gradAscent，实际上为gradDescent  
> 因为 $ w = w - \alpha \nabla{w}{f(w)} = w - \alpha (h-y) = w + \alpha(y-h)  $ 因此个人觉得直接说gradDescent更加清晰易懂，毕竟我们的目标是代价最小!


参考资料:

1. <http://eric-yuan.me/logistic-regression/>
2. <https://en.wikipedia.org/wiki/Gradient_descent>
3. Machine Learning In Action
