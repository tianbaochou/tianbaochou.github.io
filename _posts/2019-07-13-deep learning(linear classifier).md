---
layout: post
title: 线性分类器
date: 2019-07-13 20:10
description: 本专题将记录常用的线性分类器
tag: deep learning
---


# **1. 背景**

俗话说，好记性不如烂笔头，最近深有体会，很多深度学习和机器学习的基础知识，许久不回顾，就容易把这个公式，梯度计算等给忘记了。
尤其秋招快来了，我这个半个开发，半个CV很是焦虑。C++是大学的初恋，到现在学了2年的深度学习（主要是CV），Python用多了发现Python这种语言
对C++er来说更加容易表达核心思想，这也就容易把C++的很多东西给忘了。接下来的笔记中主要借助numpy来写代码段，可能也会偶尔回顾一下C++，毕竟C++才是世界上最牛逼的语言。哈哈哈！！

本系列笔记主要是记录这两年里面自己学习的心得和一个来自半吊子开发者的视角考虑问题，如果大家有缘恰好看到此博客，发现很多理解跟你们很像，或者很鄙视我的理解，敬请留个言，大家讨论一下。为了使自己每天能够有时间写一点博客，力求该系列博客精简（也可能就一个公式推导，哈哈）

# **1. 线性分类器**

More Simpler More Better! 当你发现某个问题可以用简单的方法解决时，你应该偷着乐，因为越简单粗暴的方法，你越能够深刻理解它，而不至于在半夜因为某些不可描述的状态出现而惊醒！


假设我们有$N$训练样本和标签 $(x_{1}, y_{1}, (x_{2}, y_{2}), \cdots (x_N, y_N)$， $y$ 总共有$C$种可能，即$C$个类。则线性分类器
的目标就是找到一个分类模型，其模型参数为$\theta = (w,b)$，使得对样本$x_{i}$有，$f_{i} = w^{T}x + b$，其中$f_{i}$我们不妨称为样本$x_{i}$对应$C$个类的分数向量$[f_{i}^{1}, \cdots, f_{i}^{C}]$，然后根据这个**分数向量判断其具体属于那一个类**。

那么怎么根据这个分数值判断类别呢，这就是分类器干的事情啦，最简单的分类器，当然就是

$\hat{y_{i}} = argmax(f_{i})$, 就是取分数值最大的那个作为类别。 不难编写其分类代码如下：

```python
  def predict(self, X):
        """
    Use the trained weights of this linear classifier to predict labels for
    data points.
    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
        y_pred = np.zeros(X.shape[0])                                            
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred
```

怎么找到最佳的$W$? 梯度下降呀，好继续写代码！

```python
def gradient_descent(?){

}

```

等等！梯度下降，沿着梯度的方向下降，那么怎么算梯度? 这就得先有个需要优化的目标函数（一般就是最小化损失函数）我们才能沿着优化的方向计算梯度。所以这里就需要先引入损失函数了。

## **1.1 线性SVM分类器**

在老早之前记录过李航统计学习方法中SVM公式推导过程（当然时不时还得再复习复习）。这里的SVM不同于之前的，我们不使用加入拉格朗日乘子->构造对偶式->引入KKT条件->SMO(最小序列化迭代)这个优化过程，而使用多分类SVM损失函数，又称为*Hinge Loss*.

### **Hinge Loss**

$$
L_{i} = \sum_{j \neq y_{i}}\max{(0, f_{j} - f_{y_i} + \Delta)} 
$$

从公式上可以看出，Hinge Loss的思想是要让每一个输入$x_i$对应的正确类别的模型分数值比其他那些不正确的类别分数值高出一个固定的margin $\Delta$. 将上面的$w,b$代入：

$$
L_{i}  = \sum_{j \neq y_{i}}\max{(0, w^{T}_{j}x_{i} - w^{T}_{y_i}z_i + \Delta)}
$$

其中$w_j$为w矩阵的第$j$列。 则平均损失值为：

$$

L = \frac{1}{N} \sum_{l}L_i + \lambda R(w) 

$$

其中$R(w)$为正则项，在这里的主要作用是约束$w$的值，使得最有$w$的值能够确定：这里我们取$L_2$范数

$$
R(w) = \sum_{k}\sum_{l}W_{k,l}^{2}
$$

>假设存在$w$使得$L = 0$，则$\beta w, \beta > 1$，也可以使得$L=0$。而通过限定$w$的值，可以使得$w$值在一定范围内。 例如，输入向量为$x=[1,1,1,1]$，$w_1=[0.25,0.25,0.25,0.25]$, $w_2=[1.0,0,0,0]$,则$w_{1}^{T}x=w_{2}^Tx = 1$, 通过$L_2$范数，模型更加倾向于选择$w_1$, 从这里也可以看出来，$L_2$正则化倾向于选择那些数值小，但可以分散在每个维度的权重$w$，而不是选择稀疏的权重！

好了，现在让我们来算损失函数的梯度吧!

$$

\begin{aligned}
\Delta w_{j} L_i &=& I(w_{j}^Tx_i - w^{T}_{y_i}x_i + \Delta > 0) x_i \\
\Delta w_{y_i} L_i &=& -\sum_{j \neq y_i}I(w_{j}^{T}x_i - w_{y_i}^{T}x_i + \Delta > 0)x_i
\end{aligned}

$$

其中$I$为激励函数。好了现在让我们来写代码把

```python
def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributors_count = 0 # 用于累计margin>0,
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # j != y_i 部分
                dW[:, j] += X[i]
                loss_contributors_count += 1
        # y_i部分
        dW[:, y[i]] += (-1) * loss_contributors_count * X[i]
    loss += num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dw += 2*reg*W

```


### **Softmax Loss**

+ Softmax function

$$
prob_{y_i} = \frac{e^{f_{y_i}}}{\sum_{j}^{C}e^{f_j}} 
$$

这个公式很容易可以看出代表的是一个概率值，即预测为$y_i$的概率值。

+ Softmax Loss / Cross-entropy Loss

$$
\begin{aligned}
L_{i} = -\log \frac{e^{f_{y_i}}}{\sum_{j}^{C}e^{f_j}} = -f_{y_i} + \log(\sum_{j}^{C}e^{f_j}) \\

L = \frac{1}{N} \sum_{i} f_{y_i} + \log(\sum_{j}^{C}e^{f_j})

\end{aligned}

$$

从信息学角度来看，**交叉熵(cross-entropy)** 实际上是评估真实分布$p$和预测分布$q$之间的差异程度：

$$
H(p,q) = -\sum_{x} p(x)\log(q(x))
$$

因此可以看出，Softmax分类器实际上是最小化真实分布和预测分布之间的交叉熵： 因为对于一个样本而言，其真实的类别是确定的，我们将$y$，用one-hot编码，例如有C个类，$x_i$对应的类别是$y_i$,则$y_i$的one-hot编码为$\Theta(y_i) =  [0,0,\cdots, 1, \cdots,0, 0]$,其中有且仅有一个1,且位置在第$y_i$位上。这样$\Theta(y_i)$实际上就相当于一个分布，只不过很特殊，除了一个峰外，其他都是0. 


更进一步，我们知道KL(Kullback-Leibler divergence)散度描述的就是两个分布之间的差异程度，那么:

$$
D_{KL}(p||q) = H(p,q) - H(p) \rightarrow
H(p,q) = H(p||q) - H(p)
$$

其中，$H(p)$是确定的，**因此优化交叉熵实际上等价于优化KL散度**. 

### **最小化交叉熵损失函数相当于最大化似然函数**

这个可能不是很好理解，前面我们提到，Softmax代表一个概率值，即

$$
P(y_i \mid x_i; w) = \frac{e^{f_{y_{i}}{\sum_{j}^{C} e^{f_j}}
$$

那么上面最小化交叉熵即为最小化**正确类别的$-\log$似然函数**，也就相当于最大化其似然函数。
如果我们加入正则项$R(w)$,则相当于在权重矩阵$w$中加入了**高斯先验**，这样整个损失函数就可以解释为在高斯先验下求最大后验概率MAP


> 关于数值稳定

一般在计算上面概率值时为了防止指数爆炸，会在softmax function上增加一个参数:

$$
\frac{e^{f_{y_{i}}}}{\sum_{j}e^{f_j}} = \frac{Ce^{f_{y_{i}}}}{C\sum_{j}e^{f_j}} = \frac{e^{f_{y_{i}}+logC}}{\sum_{j}e^{f_j}+logC}
$$

一般我们取$logC=-\max {f_{y_i}}$。现在我们可以来算算交叉熵损失函数的梯度:

$$

\begin{aligned}

\Delta w_j L_i = \frac{1}{N} (-x_i + \frac{e^{f_{y_i}}x_{i}^T}{\log \sum_{j}e^{f_j} + \log C}) \quad if \quad j = y_i \\

\Delta w_j L_i = \frac{1}{N} \frac{e^{f_{y_i}}x_{i}^T}{\log \sum_{j}e^{f_j} + \log C} \quad if \quad j \neq y_i


\end{aligned}

$$

好了，现在让我们看看代码如何写？

```python
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = y.shape[0]
  f = X.dot(W)
  norm = np.exp(f - f.max(axis=1, keepdims=True)) # 数值稳定
  D, C = W.shape
  N, _ = X.shape

  # 先计算loss
  for i in range(N): # For each example
      loss += -f[i, y[i]] + np.log(norm[i, :].sum())
  loss /= N
  loss += reg*(W*W).sum()

  # 后计算梯度
  for j in range(C):  # For each classes
      for i in range(N):
          sign = 1 if j == y[i] else 0
          dW[:, j] += (norm[i, j] / norm[i, :].sum() - sign)*(X[i, :].T) # Dx1
  dW /= N
  dW += 2*reg*W

  return loss, dW

```



