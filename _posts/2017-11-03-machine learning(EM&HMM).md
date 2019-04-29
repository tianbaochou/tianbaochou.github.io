---
layout: post
title: 从EM算法到HMM算法
date: 2017-04-09 20:10
description: 本专题将解析EM算法背后的推导过程以及隐马尔科夫问题
tag: machine learning
---

# **1. 背景**

&ensp;说实话,复杂的概率模型一直是我比较反感的,原因无非是理解起来比较难,而且算法的推导看上去太难.
李航[[1]](#id1)博士的$<<统计学习方法>>$后几章都是概率推导问题,前几个月时间我书看完第一遍还是感觉很难理解,
这段时间刚好接触了RNN+HMM模型,因此想借此把EM到HMM算法搞通.因此,本章节绝大部分都是李博士书上
的例子,但是会把该书中一些跳跃性比较强的推导公式部分,跟大家解释一下!

# **2. EM算法**

# **2.1 引入**

&ensp;EM算法在1977年由Arthur Dempster, Nan Laird和Donald Rubin提出并解释.它全称为
Expectation-Maximization,即**期望最大**算法. 原论文[2]中的例子关于多项式分布,理解起来比较
困难,我这里就以李博士书中的例子来引入:

**描述**

&ensp;假设有三枚硬币,分别记为A,B,C.这些硬币正面出现的概率分别为: $\pi$, $p$, $q$.进行如下
抛硬币试验:  

1. 先抛硬币A,根据其结果选出硬币B或者硬币C  
2. 若A为正面,选B,否则选C,抛出选择的硬币
3. 记录硬币抛出的结果: 正面记为1,反面记为0
4. 独立重复n次试验

观测结果如下(假设n=10):

![图片1](/images/posts/machine learning/EM&HMM/1.png)

&ensp;假设只能观测到抛硬币的结果,不能观测抛硬币的过程,那么如何估计三硬币正面出现的概率,  
即三硬币模型的参数?  

&ensp;三硬币模型可以写作:  

$$
P(y\mid \theta) = \sum \limits_{Z} P(y,z\mid \theta) = \sum \limits_{z}P(z\mid \theta)P(y\mid z,\theta) \\
            = \pi p^{y}(1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}\tag{5-1}
$$

&ensp;这里随机变量y是观测变量,表示一次试验观测的结果是1或者0;随机变量z是隐变量,表示未观测到的抛硬币A的结果;
$\theta = (\pi, p, q)$是模型参数.

若将观测数据表示未$Y = (Y_1, Y_2, \cdots, Y_n)^T$,未观测数据表示$Z = (Z_1, Z_2, \cdots, Z_n)^T$,
则观测数据的似然函数为:

$$

P(Y\mid \theta) = \sum \limits_{Z} P(Z\mid \theta)P(Y\mid Z,\theta) \tag{5-2}

$$

对应到三硬币问题,即:

$$

P(Y\mid \theta) =  \prod \limits_{j=1}^{\pi}[\pi p^{y_j}(1-p)^{1-y_j} + (1-\pi)q^{y_j}(1-q)^{1-y_j}] \tag{5-3}

$$

&ensp; 求模型参数$\theta = (\pi, p, q)$的极大似然估计,即:

$$
\hat\theta = arg \max \limits_{\theta} logP(Y\mid \theta) \tag{5-4}
$$

&ensp; 这个似然函数非凸,因此没有解析解,但是可以通过迭代的方法求解.
**EM算法就是可以用于求解这个问题的一种迭代算法**

下面我们先给出对应于本问题的EM算法的求解公式,给大家一个直观的理解,具体证明过程,大家可以在后面讲
到*Q函数*{: style="color: red"}的时候,再自行推导 :smile:

**EM算法步骤:**{: style="color: red"}

1. 选取参数的初值,记作: $\theta^{0} = (\pi^{0}, p^0, q^0)$.然后通过下面的步骤迭代计算参数
的估计值,直至收敛为止,第i次迭代参数的估计值为$\theta^i = (\pi^i, p^i, q^i)$,EM算法的第i+1次迭代如下:


1. **E步**:
    * 计算在模型参数$\pi^i, p^i, q^i$下观测数据$y_i$来自抛硬币B的概率:

    $$

      \mu^{i+1} = \frac{\pi^{i} (p^i)^{y_j}(1-p^{i})^{1-y_j}}{\pi^{i}(p^i)^{y_j}(1-p^i)^{1-y_j} +
      (1-\pi^i)(q^i)^{y_j}(1-q^i)^{1-y_j}} \tag{5-5}

    $$   


1. **M步**:
    * 计算模型参数新的估计值:

      $$

      \begin{aligned}
      \pi^{i+1} & =  \frac{1}{n} \sum \limits_{j=1}^{n} \mu_{j}^{i+1} \\
      p^{i+1} & = \frac{\sum \limits_{j=1}^{n} \mu_{j}^{i+1}y_j}{\sum \limits_{j=1}^{n} \mu_{j}^{i+1}} \\
      q^{i+1} & =  \frac{\sum \limits_{j=1}^{n}(1 - \mu_{j}^{i+1})y_j}{\sum \limits_{j=1}^{n}(1-\mu_{j}^{i+1})}
      \end{aligned}\tag{5-6}

      $$

如果我们回到前面的三硬币例子,假设模型参数的初值为: \{$\pi^{0} = 0.5, p^0 = 0.5, q^0 = 0.5$\}

当$y_j = 1$与$y_j=0$时,均有$\mu_{j}^{1} = 0.5,\quad j=1,2,\cdots,n$, 并得到$\{\pi^{1}=0.5, p^1=0.6, q^1=0.6\}$.
同理$\mu_{j}^{2}=0.5, j=1,2,\cdots,10$,并得到:$$\{\pi^2 = 0.5, p^2=0.6, q^2=0.6\}$$.
于是得到模型参数$\theta$的极大似然估计:

$$
\{\hat{\pi} = 0.5, \hat{p} = 0.6, \hat{q} = 0.6\}
$$

若我们选取初值为: $$\{ \pi^0=0.4, p^0=0.6,q^0=0.7\}$$ ,则最后得到的模型参数的极大似然估计为:
$$\{ \hat{\pi}=0.4064, \hat{p}=0.5368,
\hat{q} = 0.6432 \}$$

从上面可以看出**EM算法选择不同的初值可能得到不同的参数估计值**{: style="color: red"}


## **2.2 EM算法的导出**

&ensp;上面我们从三硬币问题引入了EM算法,让大家有个直观的理解,现在我们从似然函数出发,推导EM算法的
由来已经Q函数的形式.


### **2.2.1 似然函数**

&ensp;我们在面对含有隐变量的概率模型,目标是**极大化**观测数据Y关于参数$\theta$的对数似然函数,即
最大化:

$$

\begin{aligned}
L(\theta) & = log P(Y\mid \theta) \xrightarrow{全概率公式} \log \sum \limits_{Z}P(Y,Z\mid \theta)  \\
A          & = log \sum \limits_{Z} P(Y\mid Z,\theta)P(Z\mid \theta)
\end{aligned}\tag{5-7}

$$

### **2.2.2 求解**

&ensp; 前面我们提到似然函数非凸,因此没有解析解.实际上EM算法是通过不断的迭代逐步近似极大化$L(\theta)$,
假设在第i次迭代后$\theta$的值为$\theta^i$,我们希望新估计值$\theta$能使$L(\theta)$增加,即:
$L(\theta) > L(\theta^i)$,并逐步达到最大值,为此我们可以考虑做两者的差:  

$$

\begin{aligned}
L(\theta) - L(\theta^i) & = log(\sum \limits_{Z} P(Y\mid Z,\theta)P(Z\mid \theta)) - logP(Y\mid \theta^i) \\
L(\theta) - L(\theta^i) & = log\left( \frac{\sum \limits_{Z}P(Z\mid Y,\theta^i)P(Y\mid Z,\theta)P(Z\mid \theta)}
                      {P(Z\mid Y,\theta^i)}\right) - log P(Y\mid \theta^i)  \\
 & \geq \sum \limits_{Z}P(Z\mid Y,\theta^i) \log \frac{P(Y\mid Z)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)} - log P(Y\mid \theta^i) \\

 又因为: log P(Y\mid \theta^i) &= \sum \limits_{Z}P(Z\mid Y,\theta^i)\log P(Y\mid \theta^i),故 \\

 & = \sum \limits_{Z} P(Z\mid Y,\theta^i) \log \frac{P(Y\mid Z,\theta)P(Z\mid \theta)}
                {P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}

\end{aligned}\tag{5-8}

$$

&ensp; 式(5-8)中不等式是由Jesen不等式得到的:

$$

log \sum \limits_{j}\lambda_{j}y_{j} \geq \sum_{j}\lambda_{j}\log y_{j} \\
其中,\lambda_j \geq,\quad \sum \limits_{j}{\lambda_j} = 1 \\
\tag{5-9}
$$

&ensp;若令
$$
B(\theta, \theta^i) = L(\theta^i) + \sum \limits_{Z}P(Z\mid Y,\theta^i)\log
\frac{P(Y\mid Z,\theta)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}
$$

则$L(\theta) \geq B(\theta,\theta^i)$, 即$B(\theta,\theta^i)$是$L(\theta)$的一个下界,而且由上面知
$L(\theta^i) = B(\theta^i, \theta^i)$.因此任何使$B(\theta, \theta^i)$增大的$\theta$,
也可以使$L(\theta)$增大,为了使$L(\theta)$尽可能大,可以选择**$\theta^{i+1}$使$B(\theta,\theta^i)$达到极大**{: style="color: red"}.即:

$$  
\begin{aligned}
\theta^{i+1} & = arg \max \limits_{\theta}B(\theta, \theta^i) \\
&= arg \max \limits_{\theta}\left( L(\theta^i) + \sum \limits_{Z} P(Z\mid Y,\theta^i)log
     \frac{P(Y\mid Z,\theta)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}
     \right) \\
&= arg \max \limits_{\theta} \left( \sum \limits_{Z} P(Z\mid Y,\theta^i)log P(Y\mid Z,\theta)P(Z\mid \theta)  \right)
   + L(\theta^i) - \sum \limits_{Z}P(Z\mid Y,\theta^i)log P(Z\mid Y,\theta^i)P(Y,\theta^i) \\     
&= arg \max \limits_{\theta} \left( \sum \limits_{Z} P(Z\mid Y,\theta^i)log P(Y\mid Z,\theta)P(Z\mid \theta)  \right)
\end{aligned}\tag{5-10}
$$


&ensp; 其中,$$L(\theta^i)$$ 和 $$ \sum \limits_{Z}P(Z\mid Y,\theta^i) P(Z \mid Y,\theta^i)P(Y \mid \theta^i) $$ 对于 $$\theta$$ 来说是常数项故可以消去.


### **2.2.3 引出Q函数**

&ensp;上面(5-10)式中,我们有:

$$

arg \max \limits_{\theta}\left( \sum \limits_{Z} P(Z \mid Y,\theta^i)log P(Y,Z \mid \theta)  \right)

$$

&ensp; 为此我们定义**Q函数**为:

$$
Q(\theta, \theta^i) =  \sum \limits_{Z} P(Z \mid Y,\theta^i)log P(Y,Z \mid \theta) \tag{5-11}
$$

### **2.2.4 EM算法**

&ensp;有了Q函数,我们就可以写出一般的EM算法步骤了:

**输入:**

观测变量数据Y,隐变量数据Z,联合分布 $P(Y,Z \mid \theta)$ ,条件分布 $P(Z \mid Y,\theta)$

**输出:**

模型参数 $\theta$

**1. 初始化:**

选择参数的初值$\theta^0$,开始迭代

**2. E步:**

记$\theta^i$为第i次迭代参数$\theta$的估计值,在第i+1次迭代的E步,计算:

$$
\begin{aligned}
Q(\theta, \theta^i) & = E_{Z} [log P(Y,Z \mid \theta)\mid Y,\theta^i] \\
& = \sum \limits_{Z} log P(Y, Z \mid \theta)P(Z \mid Y, \theta^i)
\end{aligned}\tag{5-12}
$$

> 其中,$ E_{Z} [log P(Y,Z \mid \theta)\mid Y,\theta^i] $ 表示**完全数据**
$(Y,Z)$ 的对数似然函数 $log P(Y,Z\mid \theta)$ 关于在给定观测数据Y和当前参数 $\theta^{i}$ 下
对未观测数据Z的条件概率分布 $P(Z \mid Y,\theta^i)$ 的**期望**.也就是这里的**期望实际上是Q函数的一种概率统计学上的解释**

**3. M步:**

求使得$Q(\theta,\theta^i)$极大化的$\theta$,确定第$i+1$次迭代的参数估计值$\theta^{i+1}$:

$$
\theta^{i+1} = arg \max \limits_{\theta}Q(\theta, \theta^i) \tag{5-13}
$$

重复第2步和第3步,直至收敛.

上面的迭代过程在i时刻我们可以观察如下图:

![img2](/images/posts/machine learning/EM&HMM/2.png)

**EM算法正是通过不断求解下界的极大值,以此逼近对数似然函数的极大值来解决问题的!**{: style="color: blue"}


## **3. HMM算法**

### **3.1 引入**

&ensp; 为了方便大家理解隐马尔科夫模型(HMM),我们暂时放下HMM,先看看马尔科夫模型到底是干嘛用的!
为此,我将简单介绍一下马尔科夫模型和马尔科夫链(Markov Chains)

#### **3.1.1 马尔科夫链**

> 以游戏为例,任何一款游戏,其移动完全由骰子决定,则其移动序列是一个马尔科夫链又称为吸引马尔科夫链(Absorbing
Markov Chains). 这和牌类游戏如blackjack相比是不同的. 因为打牌时,我们可以根据已经打出**所有的牌**{: style="color: red"}来决定我们下一步
我们要出的牌,而在骰子类游戏中,其下一步的状态仅由当前骰子的抛掷结果决定和之前的结果无关.

下面举两个简单的满足马尔科夫链的例子:

**eg1: Random Walk**

&ensp; 考虑有一只蚂蚁在一条线上移动,其向左或者向右移动一位的概率完全由当前位置x的值决定:  

$$
P_{move\; left} = \frac{1}{2} + \frac{1}{2}(\frac{x}{c+\vert x \vert}) \\
P_{move\; right} = 1 - P_{move\; left}
$$

&ensp; 其中c为一个大于0的常数.

现在假设c等于1, 且当前的位置为如下图5:

![图3-1-1](/images/posts/machine learning/EM&HMM/3.png)

即$$x = \{-2, -1, 0, 1, 2\}$$,则向左移动的概率分别为$$\{\frac{1}{2}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, \frac{5}{6}\}$$
有上面知,蚂蚁的移动概率仅与当前的状态有关,和之前的任何状态没有任何关系,因此它满足马尔科夫链.

**eg2: Weather Predict**

&ensp; 假设天气不是晴天就是阴天,给定今天的天气,明天的天气状况由下面一个状态转移矩阵P决定:

$$

\begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix}
$$

&ensp;矩阵P代表,今天若是晴天,则明天为晴天的概率为0.9,为阴天的概率为0.1; 今天若是雨天,则明天
为晴天和雨天的概率各为0.5. 即$P_{ij}$代表:若当前的类型为i,则其下一次的类型为j. 易知,P中的每一
行之和为1.

现在假设第一天的为晴天,用一个向量表示为: $x^{(0)} = [1 \quad 0]$,代表晴天概率为1, 雨天概率为0.
根据状态转移矩阵P,可以很容易得到第二天的天气概率情况:

$$
x^{(1)} = x^{(0)}P = [1 \quad 0] \begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix} = [0.9 \quad  0.1]
$$

&ensp;则明天有90%的概率为晴天,第三天的情况为:


$$

x^{(2)} = x^{(1)}P = [0.9 \quad 0.1] \begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix} = [0.86 \quad 0.14]

$$

则第n天的情况如下:

$$
x^{(n)} = x^{(n-1)}P \\
x^{(n)} = x^{(0)}P^{n}
$$

从这个例子可以看出马尔科夫链本质上很简单. 但是这里我们有一个有趣的问题是:**随着n的增大,$x^{(n)}$是会一直变化还是会趋于稳定**,即:

$$
q = \lim \limits_{x \to n} x^{(n)}
$$

q的值的变化情况. 现在假设上面的极限存在,称q为稳态向量(steady state vector).由[[**定理1**]](https://www.math.drexel.edu/~jwd25/LM_SPRING_07/lectures/Markov.html)知:
若q为稳态向量,则$Pq = q$,即q不受状态转移矩阵P的影响.再根据[[**定理2**]](https://www.math.drexel.edu/~jwd25/LM_SPRING_07/lectures/Markov.html)知,每个状态转移矩阵P均有一个
为1的特征值.则以上面例子做说明如下:

$$
\begin{aligned}

P &= \begin{bmatrix}
    0.9 & 0.1 \\
    0.5 & 0.5
    \end{bmatrix} \\
qP &= q \quad (q \; is \; unchanged \; by \; P) \\
   &= qI \\
q(P-I) &= 0 \\
      &= q\left(
          \begin{bmatrix}
          0.9 & 0.1 \\
          0.5 & 0.5
          \end{bmatrix}
           \right)    \\
      &= q  \begin{bmatrix}
            -0.1 & 0.1 \\
            0.5 & -0.5
            \end{bmatrix} \\

\begin{bmatrix} q_1 \\ q_2 \end{bmatrix} \begin{bmatrix}
      -0.1 & 0.1 \\
      0.5 & -0.5
      \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}

\end{aligned}
$$

故$-0.1q_1 + 0.5q_2 = 0$,联合$q_2 + q_2 = 1$,得:

$$
\begin{bmatrix} q_1 \\ q_2 \end{bmatrix} = \begin{bmatrix} 0.833 \\ 0.167 \end{bmatrix}
$$

&ensp;即从长远看来,有83.3%天的概率为晴天.

> 上面仅有两个状态,因此可以这样解,当状态大于3时,需要对P矩阵进行特征值分解来分析是q是否存在!



### **3.2 描述**

&ensp; 上面简单介绍了马尔科夫过程,现在我们来介绍一下隐马尔科夫模型.
隐马尔科夫模型(Hidden Markov Model)被认为统计马尔科夫模型在马尔科夫过程中包含未观测状态
的一种形式.隐马尔科夫模型可以被认为是最简单的一个[动态贝叶斯网络](https://en.wikipedia.org/wiki/Dynamic_Bayesian_network),
其背后的数学模型由L.E.Baum和其同事共同建立. **实际上,最早和HMM非常相关的工作是在Ruslan L.Stratonvich提到的前向-后向过程中
描述的内容.**

&ensp; 隐马尔科夫模型,在增强学习和诸如语音识别,手写文字识别,手势识别等模式识别中应用广泛.

&ensp; 考虑一个例子: 假设有一个房间,里面有一个不可见的鬼. 房间中包含盒子$X_1, X_2, \cdots$,每一个盒子中包含一些球,每个球标号为$y_1, y_2, \cdots $. 鬼每次随选择一个盒子,并随机从中取出一个球,并将球放到一个传送带上. 观察者可以观测在传送带上的球,但是不可以看到鬼每次选择的盒子. 现在假设
鬼选择盒子和球的策略如下:

+ 在某个花盆中选择的第n个球仅取决于第n-1个球的选取和一个随机数
+ 下一次花盆的选择仅仅直接依赖于本次花盆的选择，和上一次无关

从上面的策略中可以看出,它是**满足马氏过程(Markov Process)的**,其过程如下图:

![img4](/images/posts/machine learning/EM&HMM/4.png)

上图中各符号表示如下:

$$
\begin{aligned}
X & \rightarrow states \\
y & \rightarrow possible \; observation \\
a & \rightarrow state \; transition \; probabilities \\
b & \rightarrow output \; probabilities
\end{aligned}
$$

&ensp;上面的马氏过程本身是无法观测的,我们只能通过观测带标签的球的序列. 回到上面的例子,我们可以
看到球$y_1,\; y_2, \; y_3, \; y_4$,但是我们并不知道每一个球是由哪个花盆来的.


### **3.2.1 描述**

&ensp; 有了上面的基础,现在我们正式进入HMM的世界.
隐马尔科夫模型由下面三个部分组成:

+ 初始概率分布
+ 转移状态概率分布
+ 观测概率分布

隐马尔科夫模型的**形式定义**如下:  

&ensp; 设 **Q** 是所有可能的状态集合,**V** 是所有可能的观测的集合.  

$$
Q = \{q_1, q_2, \cdots, q_N\}, \quad V = \{v_1, v_2, \cdots, v_M\}
$$

其中,N是可能的状态数,M是可能的观测数. I是长度为T的状态序列,O是对应的观测序列.  

$$
I = (i_1, i_2, \cdots, i_T), \quad O = (o_1,o_2, \cdots, o_T)
$$

&ensp; **A** 是状态转移概率矩阵:

$$
A = \begin{bmatrix}a_{ij}\end{bmatrix}_{N\times N}
$$

其中,

$$
a_{ij} = P(i_{t+1} = q_j \mid i_t = q_i), \quad i=1,2,\cdots,N;\;j=1,2,\cdots,N
$$

是在时刻t处于状态$q_i$的条件下在时刻t+1转移到状态$q_j$的概率.

&ensp; **B** 是观测概率矩阵:

$$
B = \begin{bmatrix} b_{j}(k) \end{bmatrix}_{N\times M}
$$

其中,

$$

b_j(k) = P(o_t=v_k \mid i_t = q_j), \quad k=1,2,\cdots,M;\; j= 1,2,\cdots,N

$$

是在时刻t处于状态$q_j$的条件下生成观测$y_k$的概率.

&ensp; $\pi$是初始状态概率向量:  

$$
\pi = (\pi_{i})
$$

其中,

$$
\pi_{i} = P(i_1 = q_i), \quad i=1,2,\cdots, N
$$

是在时刻t=1处于状态$q_i$的概率.为了方便查看,我们将其列为表:

| 参数 | 解释     |
| :-------------: | :-------------: |
| $$Q = \{q_1,q_2,\cdots,q_N\}$$       |  所有可能的状态集合       |
| $$V = \{v_1,v_2,\cdots,v_M\}$$       |  所有可能的观测的集合     |
| $$A = \begin{bmatrix}a_{ij}\end{bmatrix}_{N\times N}$$ | **状态转移概率矩阵** |
| $$a_{ij} = P(i_{t+1} = q_j \mid i_t = q_i)$$ | 在时刻t处于状态$q_i$的条件下在时刻t+1转移到状态$q_j$的概率 |
| $$B = \begin{bmatrix} b_{j}(k) \end{bmatrix}_{N\times M}$$ | **观测概率矩阵** |
| $$b_j(k) = P(o_t=v_k \mid i_t = q_j)$$ | 在时刻t处于状态$q_j$的条件下生成观测$y_k$的概率|
| $\pi$ | **初始状态概率向量** |
| $$ \pi_{i} = P(i_1 = q_i), \quad i=1,2,\cdots, N $$ | 在时刻t=1处于状态$q_i$的概率 |


&ensp; 由上面知,隐马尔科夫模型由初始状态概率向量$\pi$,状态转移矩阵A和观测概率矩阵B决定.
$\pi$和A决定状态序列,B决定观测序列. 因此,隐马尔科夫模型$\lambda$可以用三元符号表示,即:

$$
\lambda = (A,B,\pi)
$$

**隐马尔科夫模型的两个基本假设**

&ensp; 从隐马尔科夫模型的定义可以看出,隐马尔科夫模型做了两个基本假设:

1. 齐次马尔科夫性假设:

    + 假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一时刻的状态,与其他时刻的状态和观测无关,也和时刻t无关:

        $$
        P(i_t \mid t_{t-1}, o_{t-1}, \cdots, i_1,o_1) = P(i_t \mid i_{t-1}),\; t=1,2,\cdots,T \tag{5-14}
        $$

1. 观测独立性假设:

    + 假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态,与其他观测及状态无关:

        $$
        P(o_t \mid i_T, i_{T-1}, o_{T-1}, \cdots, i_{t+1}, o_{t+1}, i_{t},i_{t-1},o_{t-1},\cdots,i_1,o_1) = P(o_t \mid i_t) \tag{5-15}
        $$

下面我们举个例子来说明问题:

**eg1:**

&ensp; Alice和Bob是两个住在不同地方的朋友, 他们通常每天都通过电话讨论今天各自要干什么. Bob只喜欢做三件事:在
公园散步(walk), 购物(shop), 整理他的公寓(clean),而他选择做这三件事又受当天天气的影响. Alice没有Bob所住地区的准确天气,但是,
通过Bob每天告诉她做了什么事,她尝试推测Bob所在地区最有可能的天气.

&ensp; Alice相信天气的变化符合马尔科夫链: 这里由两种状态$Rainy$和$Sunny$,她不可以直接观察到,即对她来说是隐变量.
每一天Bob会因为天气的原因选择$$\{"walk",\; "shop",\; "clean"\}$$中的某个活动.这些活动对于Alice来说是可以知道的.
则整个系统为隐马尔卡夫模型.下面分别列出该问题的隐马尔科夫模型的**状态集合**, **观测集合**, **模型的三要素**.

**状态集合**: $$ V = \{Rainy,\; Sunny\}, \quad N = 2  $$

**观测集合**: $$ O = \{walk, \; shop, \; clean \}, \quad M = 3 $$

**模型的三要素**:

$$

\pi = \{0.6, 04 \} \\

A = \begin{bmatrix}
    0.7 & 0.3 \\
    0.4 & 0.6
    \end{bmatrix} \\

B = \begin{bmatrix}
    0.1 & 0.4 & 0.5 \\
    0.6 & 0.3 & 0.1
    \end{bmatrix}
$$

整个状态变化过程如下图:

![img5](/images/posts/machine learning/EM&HMM/5.png)


### **3.2.2 隐马尔科夫模型的三个基本问题**

1. **概率计算问题**

    + 给定模型$\lambda = (A,B,\pi)$和观测序列$O=(o_1, o_2,\cdots, o_T)$,计算在给定模型
    $\lambda$下观测序列$O$出现的概率$P(O \mid \lambda)$.

2. **学习问题**

    + 已知观测序列$O=(o_1, o_2, \cdots,o_T)$,估计模型$\lambda=(A,B,\pi)$参数,使得在该模型下
    观测序列概率$P(O\mid \lambda)最大$

3. **预测问题**

    + 已知模型$\lambda=(A,B,\pi)$和观测序列$O=(o_1,o_2,\cdots,o_T)$,求对给定观测序列条件概率
    $P(I\mid O)$最大的状态序列$I=(i_1, i_2, \cdots, i_T)$,即给定观测序列,求最有可能的对应的状态序列

下面我们将分别来讨论以上三个问题.

## **3.3. 概率计算算法**

&ensp; 为了体现前向-后向计算方法的好处,我们先来个最朴素的概率计算方法: 直接计算法.


### **3.3.1. 直接计算法**

直接计算法-顾名思义,就是通过遍历所有可能的状态序列$I=(i_1,i_2,\cdots,i_T)$,求各个状态序列$I$与
观测序列$O=(o_1, o_2,\cdots,o_T)$的联合概率$P(O,I \mid \lambda)$,然后对所有可能的状态序列求和,
得到$P(O\mid \lambda)$.

&ensp;某一时刻的状态序列$I = (i_1, i_2, \cdots, \_T)$的概率是:

$$
P(I\mid \lambda) = \pi_{i_1} a_{i_1 i_2} a_{i_2  i_3},\cdots, a_{i_{T-1}\;  i_T} \tag{5-16}
$$

&ensp;对于固定的状态序列$I=(i_1, i_2, \cdots, i_T)$,观测序列$O=(o_1,o_2,\cdots,o_T)$的概率为$P(O\mid I, \lambda)$:

$$
P(O\mid I, \lambda) = b_{i_1}(o_1)b_{i_2}(o_2)\cdots b_{i_T}(o_T) \tag{5-17}
$$

&ensp;则$O$和$I$同时出现的联合概率为:

$$

P(O,I \mid \lambda) = \frac{P(O,I, \lambda)}{P(\lambda)} = \frac{P(O\mid I,\lambda)
p(I,\lambda)}{P(\lambda)} =  P(O\mid I,\lambda)P(I\mid \lambda)\\
= b_{i_1}(o_1)b_{i_2}(o_2)\cdots b_{i_T}\pi_{i_1}a_{i_1 i_2}a_{i_2  i_3}\cdots a_{i_{T-1}\; i_T} \tag{5-18}

$$

对所有可能的状态序列求和:

$$
\begin{aligned}
P(O\mid \lambda) &= \sum \limits_{I}P(O\mid I,\lambda)P(I\mid \lambda) \\
&= \sum \limits_{i_1,i_2,\cdots, i_T}\pi_{i_1}a_{i_1 i_2}b_{i_1}(o_1)\cdots a_{i_{T-1}\;i_T}b_{i_T}(o_T)
\end{aligned}\tag{5-19}
$$

> 时间复杂度分析: 对于有N个状态数的状态集合,序列长度为T的HMM模型,总共有$N^T$种状态,每一种状态均需要2T次乘法,因此总体时间复杂度为:$O(TN^T)$.
可以看出这个时间复杂度很高,当T稍微大一点基本就无法计算了.



### **3.3.2. 前向-反向算法**

#### **3.3.2.1 前向算法**

&ensp;前向算法(或者后向算法)的基本思想就是递推求解.

**前向概率**:

到时刻$t$部分观测序列为$o_1,o_2,\cdots,o_t$且状态为$q_i$的概率:

$$
\alpha_{t}(i) = P(o_1, o_2, \cdots, o_t, i_t=q_i \mid \lambda) \tag{5-20}
$$

**前向算法的步骤**

**输入**: 隐马尔科夫模型$\lambda$,观测序列$O$;  
**输出**: 观测序列概率$P(O\mid \lambda)$

1. 初值

    + $alpha_1(i) = \pi_{i}b_{i}(o_1),\quad i=1,2,\cdots,N \tag{5-21}$

2. 递推:

    + 对$t=1,2,\cdots,T-1$:  
    $$
    \alpha_{t+1}(i) = \begin{bmatrix} \sum \limits_{j=1}^{N}\alpha_{t}(j)\end{bmatrix}b_i(0_{t+1}), \tag{5-22}
    \; i=1,2,\cdots,N
    $$

3. 终止

    + $P(O\mid \lambda) = \sum \limits_{i=1}^{N}\alpha_{T}{i} \tag{5-23}$

> 时间复杂度分析: 对于有N个状态数的状态集合,序列长度为T的HMM模型,时间复杂度为$O(TN^2)$

#### **3.3.2.2 后向算法**


**后向概率**:

给定隐马尔科夫模型$\lambda$,定义在时刻t状态为$q_i$的条件下,从$t+1$到$T$的部分观测序列为
$o_{t+1}, o_{t+1}, \cdots, o_{T}$的概率:

$$
\beta_{i}(t) = P(o_{t+1},o_{t+2},\cdots, o_T \mid i_t=q_i,\lambda)\tag{5-24}
$$

**后向算法的步骤**

**输入**: 隐马尔科夫模型$\lambda$,观测序列$O$;  
**输出**: 观测序列概率$P(O\mid \lambda)$

1. 初值

    + $\beta_{T}(i) = 1,\quad i=1,2,\cdots,N \tag{5-24}$

2. 递推

    + 对于$t=T-1, T-2, \cdots, 1$  
    $\beta_t(i) = \sum \limits_{j=1}^{N}a_{ij}b_j(o_{t+1})\beta_{t+1}(j), \quad i=1,2,\cdots,N \tag{5-25}$

3. 终止

    + $P(0\mid\lambda) = \sum \limits_{i=1}^{N}\pi_{i}b_{i}(o_1)\beta_1(i) \tag{5-26}$

实际上,利用前向概率和后向概率的定义可以将观测序列概率$P(O\mid \lambda )$统一写成:

$$
P(O \mid \lambda) = \sum \limits_{i=1}^{N} \sum \limits_{j=1}^{N}
\alpha_{t}(i)a_{ij}b_{j}(o_{t+1})\beta_{t+1}(j), \quad t=1,2,\cdots, T-1
\tag{5-27}
$$

&ensp; 当$t=1$时对应到式(5-22), $t=T-1$对应到式(5-27)


**根据前向概率和后向概率,我们可以得到一些对后面学习模型有用的概率与期望值**

1. 给定$\lambda$与观测$O$,在时刻t处于状态$q_i$的概率

$$
\begin{aligned}
\gamma_{t}(i) &= P(i_t = q_t \mid O,\lambda) \\
 &= \frac{P(i_t = q_i, O \mid \lambda)}{P(O \mid \lambda)}
 \xrightarrow{\alpha_t(i)\beta_t(i) = P(i_t=q_i,O \mid \lambda)}
 \frac{\alpha_t(i)\beta_t(i)}{\sum \limits_{j=1}^{N}\alpha_t(j)\beta_t(j)}
\end{aligned} \tag{5-28}
$$

2. 给定模型$\lambda$和观测$O$.在时刻t处于状态$q_i$且在时刻$t+1$处于状态$q_j$的概率

$$
\begin{aligned}
\epsilon_t(i,j) &= P(i_t=q_i, i_{t+1}=q_j \mid O,\lambda) \\  
&= \frac{P(i_t=q_i, i+{t+1}=q_j, O \mid \lambda)}{P(O\mid \lambda} \\
&= \frac{P(i_t=q_i, i_{t+1}=q_j, O\mid \lambda)}
{\sum \limits_{i=1}^{N} \sum \limits_{j=1}^N P(i_t=q_i,i_{t+1}=q_j,O\mid \lambda)}
\end{aligned}\tag{5-29}           \\
又P(i_t=q_i, i_{t+1}=q_j,O\mid \lambda) = \alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j) \\
\epsilon_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}
{\sum \limits_{i=1}^{N} \sum \limits_{j=1}^N(\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j))}
$$


## **3.4. 学习算法**

&ensp;下面我们来看看HMM的学习算法,根据训练数据是包括观测序列和对应的状态序列还是只有观测序列,可以分别由
监督学习与非监督学习实现.下面首先介绍监督学习再介绍非监督学习算法-**Baum-Welch(也是EM算法)**

### **3.4.1 监督学习方法**

&ensp; 假设已给训练数据包含$S$个长度相同的观测序列和对应的状态序列$$\{(O_1,I_1),(O_2,I_2),\cdots,(O_s,I_s)\}$$
,那么很容易可以想到用 **极大似然估计法(可以简单的用频率替代概率)** 来估计HMM模型的参数

1. 转移概率$a_{ij}$的估计

    + 设样本中时刻$t$处于状态$i$,时刻$t+1$转移到状态$j$的**频数**为$A_{ij}$,那么状态转移概率$a_{ij}$的估计是:

    $$
    \hat{a_{ij}} = \frac{A_{ij}}{\sum \limits_{j=1}^{N}A_{ij},\quad i=1,2,\cdots,N;\;j=1,2,\cdots,N}\tag{5-30}
    $$

2. 观测概率$b_j(k)$的估计

    + 设样本中状态为$j$并观测为$k$的频数是$B_{jk}$,那么状态为$j$观测为$k$的概率$b_j(k)$的估计是:

    $$
    \hat{b_j}(k) = \frac{B_{jk}}{\sum \limits_{k=1}^{M}B_{jk}},\quad j=1,2,\cdots,N;\;k=1,2,\cdots,M \tag{5-31}
    $$

3. 初始状态概率$\pi_i$的估计$\hat{\pi_i}$为$S$个样本中初始状态为$i$的频率



### **3.4.2 Baum-Welch算法**

上面监督学习需要使用训练数据,而人工标训练数据往往代价很高,这就要用到非监督学习算法

&ensp;假设给定训练数据只包含$S$个长度为$T$的观测序列${0_1, O_2, \cdots,O_s}$而没有对应的状态序列,目标是学习
隐马尔科夫模型$\lambda=(A,B,\pi)$的参数.HMM模型实际上是一个包含由隐变量的概率模型:

$$
P(O\mid \lambda) = \sum \limits_{I}P(O\mid I,\lambda)P(I\mid \lambda) \tag{5-32}
$$

那么如何学习这个模型的参数呢? 这就需要我们利用最开始讲的EM算法了.

1. 确定完全数据的对数似然函数

    + 所有观测数据写成$O=(o_1,o_2,\cdots,o_T)$,所有隐数据写成$I=(i_1,i_2,\cdots,i_T)$,
    完全数据是$(O,I)=(o_1,o_2,\cdots,o_T,i_1,i_2,\cdots,i_T)$.完全数据对数似然函数为:

    $$
    P(O,I\mid \lambda)
    $$

2. EM算法的E步

    + 求$Q$函数$Q(\lambda,\bar{\lambda})$

    $$
    \begin{aligned}
    Q(\lambda,\bar{\lambda}) &= E_t[\log P(O,I\mid \lambda) \mid O,\bar{\lambda}] \\
    &= \sum \limits_{I}\log P(I\mid O,\bar{\lambda})\log P(I,O\mid \lambda) \\
    &= \sum \limits_{I} \log \frac{P(I,O\mid \bar{\lambda})}{P(O\mid \bar{\lambda})}
        \log P(I,O \mid \lambda) \\
    & \xrightarrow{省去对\lambda而言的常数因子\frac{1}{P(O\mid \bar{\lambda})}}
    \sum \limits_{I}\log P(I,O\mid \bar{\lambda})P(I,O\mid \lambda)
    \end{aligned} \tag{5-33}
    $$
    其中,$\bar{\lambda}$是隐HMM参数的当前估计值,$\lambda$是要极大化的HMM参数.又  

    $$
    P(O,I \mid \lambda) = \pi_{i_1}b_{i_1}(o_1)a_{i_1 i_2}b_{i_2}(o_2)\cdots a_{i_{T-1}\; i_T}b_{i_T}(o_T)
    $$
    于是函数$Q=(\lambda, \bar{\lambda})$可以写成:

    $$
    Q(\lambda,\bar{\lambda}) = \sum \limits_{I}\log \pi_{i_1}P(O,I\mid \bar{\lambda})
        + \sum \limits_{I}\left(\sum \limits_{t=1}^{T-1} \log a_{i_1 i_t+1}\right)P(O,I \mid \bar{\lambda})
        + \sum \limits_{I}\left(\sum \limits_{t=1}^{T} \log b_{i_t}(o_t)\right)P(O,I\mid \bar{\lambda})  \tag{5-34}
    $$

2. EM算法的M步

    + 极大化$Q$函数$Q(\lambda, \bar{\lambda})$求模型参数$A,B,\pi$

    由于到极大化的参数在式5-34中单独出现在3个项中,所有只需对各项**分别极大化**.
    在极大化的过程中,分别利用$\sum \limits_{i=1}^{N}\pi_i = 1,\; \sum \limits_{j=1}^{N}a_{ij}=1,\;\sum \limits_{k=1}^{M}b_j(k)=1$
    这三个约束条件,再用**拉格朗日乘子法**,对待求参数求偏导并令结果等于0,即可解出.
    由于求导过程比较繁琐,大家如果感兴趣可以参考[5].我这里直接得出结论:

    $$
    \begin{aligned}

    a_{ij} &= \frac{\sum \limits_{t=1}^{T-1}\epsilon_{t}(i,j)}{\sum \limits_{t=1}^{T-1}}\gamma_t(i) \\
    b_j(k) &= \frac{\sum \limits_{t=1,o_t=v_k}^{T}\gamma_t(j)}{\sum \limits_{t=1}\gamma_t(j)} \\
    \pi_i  &= \gamma_1(i)    
    \end{aligned}

    $$

## **3.5. 预测算法**

HMM有两种预测算法: 近似算法与维特比算法.
由于本章篇幅太长了，且这两个算法都比较简单，这里就不介绍.


**参考文献**

<span id = "id1">[1] 李航, 统计学习方法. </span>  
[2] A. P. Dempster; N. M. Laird; D. B. Rubin. Maximum Likelihood from Incomplete Data via the EM Algorithm.  
[3] Hidden Markov model: https://en.wikipedia.org/wiki/Hidden_Markov_model.  
[4] Markov_chain: https://en.wikipedia.org/wiki/Markov_chain.  
[5] Daniel Ramage,Hidden Markov Models Fundamentals, CS229 Section Notes.  
---
layout: post
title: 从EM算法到HMM算法
date: 2017-04-09 20:10
description: 本专题将解析EM算法背后的推导过程以及隐马尔科夫问题
tag: machine learning
---


# **1. 背景**

&ensp;说实话,复杂的概率模型一直是我比较反感的,原因无非是理解起来比较难,而且算法的推导看上去太难.
李航[[1]](#id1)博士的$<<统计学习方法>>$后几章都是概率推导问题,前几个月时间我书看完第一遍还是感觉很难理解,
这段时间刚好接触了RNN+HMM模型,因此想借此把EM到HMM算法搞通.因此,本章节绝大部分都是李博士书上
的例子,但是会把该书中一些跳跃性比较强的推导公式部分,跟大家解释一下!

# **2. EM算法**

# **2.1 引入**

&ensp;EM算法在1977年由Arthur Dempster, Nan Laird和Donald Rubin提出并解释.它全称为
Expectation-Maximization,即**期望最大**算法. 原论文[2]中的例子关于多项式分布,理解起来比较
困难,我这里就以李博士书中的例子来引入:

**描述**

&ensp;假设有三枚硬币,分别记为A,B,C.这些硬币正面出现的概率分别为: $\pi$, $p$, $q$.进行如下
抛硬币试验:  

1. 先抛硬币A,根据其结果选出硬币B或者硬币C  
2. 若A为正面,选B,否则选C,抛出选择的硬币
3. 记录硬币抛出的结果: 正面记为1,反面记为0
4. 独立重复n次试验

观测结果如下(假设n=10):

![图片1](/images/posts/machine learning/EM&HMM/1.png)

&ensp;假设只能观测到抛硬币的结果,不能观测抛硬币的过程,那么如何估计三硬币正面出现的概率,  
即三硬币模型的参数?  

&ensp;三硬币模型可以写作:  

$$
P(y\mid \theta) = \sum \limits_{Z} P(y,z\mid \theta) = \sum \limits_{z}P(z\mid \theta)P(y\mid z,\theta) \\
            = \pi p^{y}(1-p)^{1-y} + (1-\pi)q^y(1-q)^{1-y}\tag{5-1}
$$

&ensp;这里随机变量y是观测变量,表示一次试验观测的结果是1或者0;随机变量z是隐变量,表示未观测到的抛硬币A的结果;
$\theta = (\pi, p, q)$是模型参数.

若将观测数据表示未$Y = (Y_1, Y_2, \cdots, Y_n)^T$,未观测数据表示$Z = (Z_1, Z_2, \cdots, Z_n)^T$,
则观测数据的似然函数为:

$$

P(Y\mid \theta) = \sum \limits_{Z} P(Z\mid \theta)P(Y\mid Z,\theta) \tag{5-2}

$$

对应到三硬币问题,即:

$$

P(Y\mid \theta) =  \prod \limits_{j=1}^{\pi}[\pi p^{y_j}(1-p)^{1-y_j} + (1-\pi)q^{y_j}(1-q)^{1-y_j}] \tag{5-3}

$$

&ensp; 求模型参数$\theta = (\pi, p, q)$的极大似然估计,即:

$$
\hat\theta = arg \max \limits_{\theta} logP(Y\mid \theta) \tag{5-4}
$$

&ensp; 这个似然函数非凸,因此没有解析解,但是可以通过迭代的方法求解.
**EM算法就是可以用于求解这个问题的一种迭代算法**

下面我们先给出对应于本问题的EM算法的求解公式,给大家一个直观的理解,具体证明过程,大家可以在后面讲
到*Q函数*{: style="color: red"}的时候,再自行推导 :smile:

**EM算法步骤:**{: style="color: red"}

1. 选取参数的初值,记作: $\theta^{0} = (\pi^{0}, p^0, q^0)$.然后通过下面的步骤迭代计算参数
的估计值,直至收敛为止,第i次迭代参数的估计值为$\theta^i = (\pi^i, p^i, q^i)$,EM算法的第i+1次迭代如下:


1. **E步**:
    * 计算在模型参数$\pi^i, p^i, q^i$下观测数据$y_i$来自抛硬币B的概率:

    $$

      \mu^{i+1} = \frac{\pi^{i} (p^i)^{y_j}(1-p^{i})^{1-y_j}}{\pi^{i}(p^i)^{y_j}(1-p^i)^{1-y_j} +
      (1-\pi^i)(q^i)^{y_j}(1-q^i)^{1-y_j}} \tag{5-5}

    $$   


1. **M步**:
    * 计算模型参数新的估计值:

      $$

      \begin{aligned}
      \pi^{i+1} & =  \frac{1}{n} \sum \limits_{j=1}^{n} \mu_{j}^{i+1} \\
      p^{i+1} & = \frac{\sum \limits_{j=1}^{n} \mu_{j}^{i+1}y_j}{\sum \limits_{j=1}^{n} \mu_{j}^{i+1}} \\
      q^{i+1} & =  \frac{\sum \limits_{j=1}^{n}(1 - \mu_{j}^{i+1})y_j}{\sum \limits_{j=1}^{n}(1-\mu_{j}^{i+1})}
      \end{aligned}\tag{5-6}

      $$

如果我们回到前面的三硬币例子,假设模型参数的初值为: \{$\pi^{0} = 0.5, p^0 = 0.5, q^0 = 0.5$\}

当$y_j = 1$与$y_j=0$时,均有$\mu_{j}^{1} = 0.5,\quad j=1,2,\cdots,n$, 并得到$\{\pi^{1}=0.5, p^1=0.6, q^1=0.6\}$.
同理$\mu_{j}^{2}=0.5, j=1,2,\cdots,10$,并得到:$$\{\pi^2 = 0.5, p^2=0.6, q^2=0.6\}$$.
于是得到模型参数$\theta$的极大似然估计:

$$
\{\hat{\pi} = 0.5, \hat{p} = 0.6, \hat{q} = 0.6\}
$$

若我们选取初值为: $$\{ \pi^0=0.4, p^0=0.6,q^0=0.7\}$$ ,则最后得到的模型参数的极大似然估计为:
$$\{ \hat{\pi}=0.4064, \hat{p}=0.5368,
\hat{q} = 0.6432 \}$$

从上面可以看出**EM算法选择不同的初值可能得到不同的参数估计值**{: style="color: red"}


## **2.2 EM算法的导出**

&ensp;上面我们从三硬币问题引入了EM算法,让大家有个直观的理解,现在我们从似然函数出发,推导EM算法的
由来已经Q函数的形式.


### **2.2.1 似然函数**

&ensp;我们在面对含有隐变量的概率模型,目标是**极大化**观测数据Y关于参数$\theta$的对数似然函数,即
最大化:

$$

\begin{aligned}
L(\theta) & = log P(Y\mid \theta) \xrightarrow{全概率公式} \log \sum \limits_{Z}P(Y,Z\mid \theta)  \\
A          & = log \sum \limits_{Z} P(Y\mid Z,\theta)P(Z\mid \theta)
\end{aligned}\tag{5-7}

$$

### **2.2.2 求解**

&ensp; 前面我们提到似然函数非凸,因此没有解析解.实际上EM算法是通过不断的迭代逐步近似极大化$L(\theta)$,
假设在第i次迭代后$\theta$的值为$\theta^i$,我们希望新估计值$\theta$能使$L(\theta)$增加,即:
$L(\theta) > L(\theta^i)$,并逐步达到最大值,为此我们可以考虑做两者的差:  

$$

\begin{aligned}
L(\theta) - L(\theta^i) & = log(\sum \limits_{Z} P(Y\mid Z,\theta)P(Z\mid \theta)) - logP(Y\mid \theta^i) \\
L(\theta) - L(\theta^i) & = log\left( \frac{\sum \limits_{Z}P(Z\mid Y,\theta^i)P(Y\mid Z,\theta)P(Z\mid \theta)}
                      {P(Z\mid Y,\theta^i)}\right) - log P(Y\mid \theta^i)  \\
 & \geq \sum \limits_{Z}P(Z\mid Y,\theta^i) \log \frac{P(Y\mid Z)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)} - log P(Y\mid \theta^i) \\

 又因为: log P(Y\mid \theta^i) &= \sum \limits_{Z}P(Z\mid Y,\theta^i)\log P(Y\mid \theta^i),故 \\

 & = \sum \limits_{Z} P(Z\mid Y,\theta^i) \log \frac{P(Y\mid Z,\theta)P(Z\mid \theta)}
                {P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}

\end{aligned}\tag{5-8}

$$

&ensp; 式(5-8)中不等式是由Jesen不等式得到的:

$$

log \sum \limits_{j}\lambda_{j}y_{j} \geq \sum_{j}\lambda_{j}\log y_{j} \\
其中,\lambda_j \geq,\quad \sum \limits_{j}{\lambda_j} = 1 \\
\tag{5-9}
$$

&ensp;若令
$$
B(\theta, \theta^i) = L(\theta^i) + \sum \limits_{Z}P(Z\mid Y,\theta^i)\log
\frac{P(Y\mid Z,\theta)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}
$$

则$L(\theta) \geq B(\theta,\theta^i)$, 即$B(\theta,\theta^i)$是$L(\theta)$的一个下界,而且由上面知
$L(\theta^i) = B(\theta^i, \theta^i)$.因此任何使$B(\theta, \theta^i)$增大的$\theta$,
也可以使$L(\theta)$增大,为了使$L(\theta)$尽可能大,可以选择**$\theta^{i+1}$使$B(\theta,\theta^i)$达到极大**{: style="color: red"}.即:

$$  
\begin{aligned}
\theta^{i+1} & = arg \max \limits_{\theta}B(\theta, \theta^i) \\
&= arg \max \limits_{\theta}\left( L(\theta^i) + \sum \limits_{Z} P(Z\mid Y,\theta^i)log
     \frac{P(Y\mid Z,\theta)P(Z\mid \theta)}{P(Z\mid Y,\theta^i)P(Y\mid \theta^i)}
     \right) \\
&= arg \max \limits_{\theta} \left( \sum \limits_{Z} P(Z\mid Y,\theta^i)log P(Y\mid Z,\theta)P(Z\mid \theta)  \right)
   + L(\theta^i) - \sum \limits_{Z}P(Z\mid Y,\theta^i)log P(Z\mid Y,\theta^i)P(Y,\theta^i) \\     
&= arg \max \limits_{\theta} \left( \sum \limits_{Z} P(Z\mid Y,\theta^i)log P(Y\mid Z,\theta)P(Z\mid \theta)  \right)
\end{aligned}\tag{5-10}
$$


&ensp; 其中,$$L(\theta^i)$$ 和 $$ \sum \limits_{Z}P(Z\mid Y,\theta^i) P(Z \mid Y,\theta^i)P(Y \mid \theta^i) $$ 对于 $$\theta$$ 来说是常数项故可以消去.


### **2.2.3 引出Q函数**

&ensp;上面(5-10)式中,我们有:

$$

arg \max \limits_{\theta}\left( \sum \limits_{Z} P(Z \mid Y,\theta^i)log P(Y,Z \mid \theta)  \right)

$$

&ensp; 为此我们定义**Q函数**为:

$$
Q(\theta, \theta^i) =  \sum \limits_{Z} P(Z \mid Y,\theta^i)log P(Y,Z \mid \theta) \tag{5-11}
$$

### **2.2.4 EM算法**

&ensp;有了Q函数,我们就可以写出一般的EM算法步骤了:

**输入:**

观测变量数据Y,隐变量数据Z,联合分布 $P(Y,Z \mid \theta)$ ,条件分布 $P(Z \mid Y,\theta)$

**输出:**

模型参数 $\theta$

**1. 初始化:**

选择参数的初值$\theta^0$,开始迭代

**2. E步:**

记$\theta^i$为第i次迭代参数$\theta$的估计值,在第i+1次迭代的E步,计算:

$$
\begin{aligned}
Q(\theta, \theta^i) & = E_{Z} [log P(Y,Z \mid \theta)\mid Y,\theta^i] \\
& = \sum \limits_{Z} log P(Y, Z \mid \theta)P(Z \mid Y, \theta^i)
\end{aligned}\tag{5-12}
$$

> 其中,$ E_{Z} [log P(Y,Z \mid \theta)\mid Y,\theta^i] $ 表示**完全数据**
$(Y,Z)$ 的对数似然函数 $log P(Y,Z\mid \theta)$ 关于在给定观测数据Y和当前参数 $\theta^{i}$ 下
对未观测数据Z的条件概率分布 $P(Z \mid Y,\theta^i)$ 的**期望**.也就是这里的**期望实际上是Q函数的一种概率统计学上的解释**

**3. M步:**

求使得$Q(\theta,\theta^i)$极大化的$\theta$,确定第$i+1$次迭代的参数估计值$\theta^{i+1}$:

$$
\theta^{i+1} = arg \max \limits_{\theta}Q(\theta, \theta^i) \tag{5-13}
$$

重复第2步和第3步,直至收敛.

上面的迭代过程在i时刻我们可以观察如下图:

![img2](/images/posts/machine learning/EM&HMM/2.png)

**EM算法正是通过不断求解下界的极大值,以此逼近对数似然函数的极大值来解决问题的!**{: style="color: blue"}


## **3. HMM算法**

### **3.1 引入**

&ensp; 为了方便大家理解隐马尔科夫模型(HMM),我们暂时放下HMM,先看看马尔科夫模型到底是干嘛用的!
为此,我将简单介绍一下马尔科夫模型和马尔科夫链(Markov Chains)

#### **3.1.1 马尔科夫链**

> 以游戏为例,任何一款游戏,其移动完全由骰子决定,则其移动序列是一个马尔科夫链又称为吸引马尔科夫链(Absorbing
Markov Chains). 这和牌类游戏如blackjack相比是不同的. 因为打牌时,我们可以根据已经打出**所有的牌**{: style="color: red"}来决定我们下一步
我们要出的牌,而在骰子类游戏中,其下一步的状态仅由当前骰子的抛掷结果决定和之前的结果无关.

下面举两个简单的满足马尔科夫链的例子:

**eg1: Random Walk**

&ensp; 考虑有一只蚂蚁在一条线上移动,其向左或者向右移动一位的概率完全由当前位置x的值决定:  

$$
P_{move\; left} = \frac{1}{2} + \frac{1}{2}(\frac{x}{c+\vert x \vert}) \\
P_{move\; right} = 1 - P_{move\; left}
$$

&ensp; 其中c为一个大于0的常数.

现在假设c等于1, 且当前的位置为如下图5:

![图3-1-1](/images/posts/machine learning/EM&HMM/3.png)

即$$x = \{-2, -1, 0, 1, 2\}$$,则向左移动的概率分别为$$\{\frac{1}{2}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, \frac{5}{6}\}$$
有上面知,蚂蚁的移动概率仅与当前的状态有关,和之前的任何状态没有任何关系,因此它满足马尔科夫链.

**eg2: Weather Predict**

&ensp; 假设天气不是晴天就是阴天,给定今天的天气,明天的天气状况由下面一个状态转移矩阵P决定:

$$

\begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix}
$$

&ensp;矩阵P代表,今天若是晴天,则明天为晴天的概率为0.9,为阴天的概率为0.1; 今天若是雨天,则明天
为晴天和雨天的概率各为0.5. 即$P_{ij}$代表:若当前的类型为i,则其下一次的类型为j. 易知,P中的每一
行之和为1.

现在假设第一天的为晴天,用一个向量表示为: $x^{(0)} = [1 \quad 0]$,代表晴天概率为1, 雨天概率为0.
根据状态转移矩阵P,可以很容易得到第二天的天气概率情况:

$$
x^{(1)} = x^{(0)}P = [1 \quad 0] \begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix} = [0.9 \quad  0.1]
$$

&ensp;则明天有90%的概率为晴天,第三天的情况为:


$$

x^{(2)} = x^{(1)}P = [0.9 \quad 0.1] \begin{bmatrix}
0.9 & 0.1\\
0.5 & 0.5
\end{bmatrix} = [0.86 \quad 0.14]

$$

则第n天的情况如下:

$$
x^{(n)} = x^{(n-1)}P \\
x^{(n)} = x^{(0)}P^{n}
$$

从这个例子可以看出马尔科夫链本质上很简单. 但是这里我们有一个有趣的问题是:**随着n的增大,$x^{(n)}$是会一直变化还是会趋于稳定**,即:

$$
q = \lim \limits_{x \to n} x^{(n)}
$$

q的值的变化情况. 现在假设上面的极限存在,称q为稳态向量(steady state vector).由[[**定理1**]](https://www.math.drexel.edu/~jwd25/LM_SPRING_07/lectures/Markov.html)知:
若q为稳态向量,则$Pq = q$,即q不受状态转移矩阵P的影响.再根据[[**定理2**]](https://www.math.drexel.edu/~jwd25/LM_SPRING_07/lectures/Markov.html)知,每个状态转移矩阵P均有一个
为1的特征值.则以上面例子做说明如下:

$$
\begin{aligned}

P &= \begin{bmatrix}
    0.9 & 0.1 \\
    0.5 & 0.5
    \end{bmatrix} \\
qP &= q \quad (q \; is \; unchanged \; by \; P) \\
   &= qI \\
q(P-I) &= 0 \\
      &= q\left(
          \begin{bmatrix}
          0.9 & 0.1 \\
          0.5 & 0.5
          \end{bmatrix}
           \right)    \\
      &= q  \begin{bmatrix}
            -0.1 & 0.1 \\
            0.5 & -0.5
            \end{bmatrix} \\

\begin{bmatrix} q_1 \\ q_2 \end{bmatrix} \begin{bmatrix}
      -0.1 & 0.1 \\
      0.5 & -0.5
      \end{bmatrix} &= \begin{bmatrix} 0 \\ 0 \end{bmatrix}

\end{aligned}
$$

故$-0.1q_1 + 0.5q_2 = 0$,联合$q_2 + q_2 = 1$,得:

$$
\begin{bmatrix} q_1 \\ q_2 \end{bmatrix} = \begin{bmatrix} 0.833 \\ 0.167 \end{bmatrix}
$$

&ensp;即从长远看来,有83.3%天的概率为晴天.

> 上面仅有两个状态,因此可以这样解,当状态大于3时,需要对P矩阵进行特征值分解来分析是q是否存在!



### **3.2 描述**

&ensp; 上面简单介绍了马尔科夫过程,现在我们来介绍一下隐马尔科夫模型.
隐马尔科夫模型(Hidden Markov Model)被认为统计马尔科夫模型在马尔科夫过程中包含未观测状态
的一种形式.隐马尔科夫模型可以被认为是最简单的一个[动态贝叶斯网络](https://en.wikipedia.org/wiki/Dynamic_Bayesian_network),
其背后的数学模型由L.E.Baum和其同事共同建立. **实际上,最早和HMM非常相关的工作是在Ruslan L.Stratonvich提到的前向-后向过程中
描述的内容.**

&ensp; 隐马尔科夫模型,在增强学习和诸如语音识别,手写文字识别,手势识别等模式识别中应用广泛.

&ensp; 考虑一个例子: 假设有一个房间,里面有一个不可见的鬼. 房间中包含盒子$X_1, X_2, \cdots$,每一个盒子中包含一些球,每个球标号为$y_1, y_2, \cdots $. 鬼每次随选择一个盒子,并随机从中取出一个球,并将球放到一个传送带上. 观察者可以观测在传送带上的球,但是不可以看到鬼每次选择的盒子. 现在假设
鬼选择盒子和球的策略如下:

+ 在某个花盆中选择的第n个球仅取决于第n-1个球的选取和一个随机数
+ 下一次花盆的选择仅仅直接依赖于本次花盆的选择，和上一次无关

从上面的策略中可以看出,它是**满足马氏过程(Markov Process)的**,其过程如下图:

![img4](/images/posts/machine learning/EM&HMM/4.png)

上图中各符号表示如下:

$$
\begin{aligned}
X & \rightarrow states \\
y & \rightarrow possible \; observation \\
a & \rightarrow state \; transition \; probabilities \\
b & \rightarrow output \; probabilities
\end{aligned}
$$

&ensp;上面的马氏过程本身是无法观测的,我们只能通过观测带标签的球的序列. 回到上面的例子,我们可以
看到球$y_1,\; y_2, \; y_3, \; y_4$,但是我们并不知道每一个球是由哪个花盆来的.


### **3.2.1 描述**

&ensp; 有了上面的基础,现在我们正式进入HMM的世界.
隐马尔科夫模型由下面三个部分组成:

+ 初始概率分布
+ 转移状态概率分布
+ 观测概率分布

隐马尔科夫模型的**形式定义**如下:  

&ensp; 设 **Q** 是所有可能的状态集合,**V** 是所有可能的观测的集合.  

$$
Q = \{q_1, q_2, \cdots, q_N\}, \quad V = \{v_1, v_2, \cdots, v_M\}
$$

其中,N是可能的状态数,M是可能的观测数. I是长度为T的状态序列,O是对应的观测序列.  

$$
I = (i_1, i_2, \cdots, i_T), \quad O = (o_1,o_2, \cdots, o_T)
$$

&ensp; **A** 是状态转移概率矩阵:

$$
A = \begin{bmatrix}a_{ij}\end{bmatrix}_{N\times N}
$$

其中,

$$
a_{ij} = P(i_{t+1} = q_j \mid i_t = q_i), \quad i=1,2,\cdots,N;\;j=1,2,\cdots,N
$$

是在时刻t处于状态$q_i$的条件下在时刻t+1转移到状态$q_j$的概率.

&ensp; **B** 是观测概率矩阵:

$$
B = \begin{bmatrix} b_{j}(k) \end{bmatrix}_{N\times M}
$$

其中,

$$

b_j(k) = P(o_t=v_k \mid i_t = q_j), \quad k=1,2,\cdots,M;\; j= 1,2,\cdots,N

$$

是在时刻t处于状态$q_j$的条件下生成观测$y_k$的概率.

&ensp; $\pi$是初始状态概率向量:  

$$
\pi = (\pi_{i})
$$

其中,

$$
\pi_{i} = P(i_1 = q_i), \quad i=1,2,\cdots, N
$$

是在时刻t=1处于状态$q_i$的概率.为了方便查看,我们将其列为表:

| 参数 | 解释     |
| :-------------: | :-------------: |
| $$Q = \{q_1,q_2,\cdots,q_N\}$$       |  所有可能的状态集合       |
| $$V = \{v_1,v_2,\cdots,v_M\}$$       |  所有可能的观测的集合     |
| $$A = \begin{bmatrix}a_{ij}\end{bmatrix}_{N\times N}$$ | **状态转移概率矩阵** |
| $$a_{ij} = P(i_{t+1} = q_j \mid i_t = q_i)$$ | 在时刻t处于状态$q_i$的条件下在时刻t+1转移到状态$q_j$的概率 |
| $$B = \begin{bmatrix} b_{j}(k) \end{bmatrix}_{N\times M}$$ | **观测概率矩阵** |
| $$b_j(k) = P(o_t=v_k \mid i_t = q_j)$$ | 在时刻t处于状态$q_j$的条件下生成观测$y_k$的概率|
| $\pi$ | **初始状态概率向量** |
| $$ \pi_{i} = P(i_1 = q_i), \quad i=1,2,\cdots, N $$ | 在时刻t=1处于状态$q_i$的概率 |


&ensp; 由上面知,隐马尔科夫模型由初始状态概率向量$\pi$,状态转移矩阵A和观测概率矩阵B决定.
$\pi$和A决定状态序列,B决定观测序列. 因此,隐马尔科夫模型$\lambda$可以用三元符号表示,即:

$$
\lambda = (A,B,\pi)
$$

**隐马尔科夫模型的两个基本假设**

&ensp; 从隐马尔科夫模型的定义可以看出,隐马尔科夫模型做了两个基本假设:

1. 齐次马尔科夫性假设:

    + 假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一时刻的状态,与其他时刻的状态和观测无关,也和时刻t无关:

        $$
        P(i_t \mid t_{t-1}, o_{t-1}, \cdots, i_1,o_1) = P(i_t \mid i_{t-1}),\; t=1,2,\cdots,T \tag{5-14}
        $$

1. 观测独立性假设:

    + 假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态,与其他观测及状态无关:

        $$
        P(o_t \mid i_T, i_{T-1}, o_{T-1}, \cdots, i_{t+1}, o_{t+1}, i_{t},i_{t-1},o_{t-1},\cdots,i_1,o_1) = P(o_t \mid i_t) \tag{5-15}
        $$

下面我们举个例子来说明问题:

**eg1:**

&ensp; Alice和Bob是两个住在不同地方的朋友, 他们通常每天都通过电话讨论今天各自要干什么. Bob只喜欢做三件事:在
公园散步(walk), 购物(shop), 整理他的公寓(clean),而他选择做这三件事又受当天天气的影响. Alice没有Bob所住地区的准确天气,但是,
通过Bob每天告诉她做了什么事,她尝试推测Bob所在地区最有可能的天气.

&ensp; Alice相信天气的变化符合马尔科夫链: 这里由两种状态$Rainy$和$Sunny$,她不可以直接观察到,即对她来说是隐变量.
每一天Bob会因为天气的原因选择$$\{"walk",\; "shop",\; "clean"\}$$中的某个活动.这些活动对于Alice来说是可以知道的.
则整个系统为隐马尔卡夫模型.下面分别列出该问题的隐马尔科夫模型的**状态集合**, **观测集合**, **模型的三要素**.

**状态集合**: $$ V = \{Rainy,\; Sunny\}, \quad N = 2  $$

**观测集合**: $$ O = \{walk, \; shop, \; clean \}, \quad M = 3 $$

**模型的三要素**:

$$

\pi = \{0.6, 04 \} \\

A = \begin{bmatrix}
    0.7 & 0.3 \\
    0.4 & 0.6
    \end{bmatrix} \\

B = \begin{bmatrix}
    0.1 & 0.4 & 0.5 \\
    0.6 & 0.3 & 0.1
    \end{bmatrix}
$$

整个状态变化过程如下图:

![img5](/images/posts/machine learning/EM&HMM/5.png)


### **3.2.2 隐马尔科夫模型的三个基本问题**

1. **概率计算问题**

    + 给定模型$\lambda = (A,B,\pi)$和观测序列$O=(o_1, o_2,\cdots, o_T)$,计算在给定模型
    $\lambda$下观测序列$O$出现的概率$P(O \mid \lambda)$.

2. **学习问题**

    + 已知观测序列$O=(o_1, o_2, \cdots,o_T)$,估计模型$\lambda=(A,B,\pi)$参数,使得在该模型下
    观测序列概率$P(O\mid \lambda)最大$

3. **预测问题**

    + 已知模型$\lambda=(A,B,\pi)$和观测序列$O=(o_1,o_2,\cdots,o_T)$,求对给定观测序列条件概率
    $P(I\mid O)$最大的状态序列$I=(i_1, i_2, \cdots, i_T)$,即给定观测序列,求最有可能的对应的状态序列

下面我们将分别来讨论以上三个问题.

## **3.3. 概率计算算法**

&ensp; 为了体现前向-后向计算方法的好处,我们先来个最朴素的概率计算方法: 直接计算法.


### **3.3.1. 直接计算法**

直接计算法-顾名思义,就是通过遍历所有可能的状态序列$I=(i_1,i_2,\cdots,i_T)$,求各个状态序列$I$与
观测序列$O=(o_1, o_2,\cdots,o_T)$的联合概率$P(O,I \mid \lambda)$,然后对所有可能的状态序列求和,
得到$P(O\mid \lambda)$.

&ensp;某一时刻的状态序列$I = (i_1, i_2, \cdots, \_T)$的概率是:

$$
P(I\mid \lambda) = \pi_{i_1} a_{i_1 i_2} a_{i_2  i_3},\cdots, a_{i_{T-1}\;  i_T} \tag{5-16}
$$

&ensp;对于固定的状态序列$I=(i_1, i_2, \cdots, i_T)$,观测序列$O=(o_1,o_2,\cdots,o_T)$的概率为$P(O\mid I, \lambda)$:

$$
P(O\mid I, \lambda) = b_{i_1}(o_1)b_{i_2}(o_2)\cdots b_{i_T}(o_T) \tag{5-17}
$$

&ensp;则$O$和$I$同时出现的联合概率为:

$$

P(O,I \mid \lambda) = \frac{P(O,I, \lambda)}{P(\lambda)} = \frac{P(O\mid I,\lambda)
p(I,\lambda)}{P(\lambda)} =  P(O\mid I,\lambda)P(I\mid \lambda)\\
= b_{i_1}(o_1)b_{i_2}(o_2)\cdots b_{i_T}\pi_{i_1}a_{i_1 i_2}a_{i_2  i_3}\cdots a_{i_{T-1}\; i_T} \tag{5-18}

$$

对所有可能的状态序列求和:

$$
\begin{aligned}
P(O\mid \lambda) &= \sum \limits_{I}P(O\mid I,\lambda)P(I\mid \lambda) \\
&= \sum \limits_{i_1,i_2,\cdots, i_T}\pi_{i_1}a_{i_1 i_2}b_{i_1}(o_1)\cdots a_{i_{T-1}\;i_T}b_{i_T}(o_T)
\end{aligned}\tag{5-19}
$$

> 时间复杂度分析: 对于有N个状态数的状态集合,序列长度为T的HMM模型,总共有$N^T$种状态,每一种状态均需要2T次乘法,因此总体时间复杂度为:$O(TN^T)$.
可以看出这个时间复杂度很高,当T稍微大一点基本就无法计算了.



### **3.3.2. 前向-反向算法**

#### **3.3.2.1 前向算法**

&ensp;前向算法(或者后向算法)的基本思想就是递推求解.

**前向概率**:

到时刻$t$部分观测序列为$o_1,o_2,\cdots,o_t$且状态为$q_i$的概率:

$$
\alpha_{t}(i) = P(o_1, o_2, \cdots, o_t, i_t=q_i \mid \lambda) \tag{5-20}
$$

**前向算法的步骤**

**输入**: 隐马尔科夫模型$\lambda$,观测序列$O$;  
**输出**: 观测序列概率$P(O\mid \lambda)$

1. 初值

    + $\alpha_1(i) = \pi_{i}b_{i}(o_1),\quad i=1,2,\cdots,N \tag{5-21}$

2. 递推:

    + 对$t=1,2,\cdots,T-1$:  
    $$
    \alpha_{t+1}(i) = \begin{bmatrix} \sum \limits_{j=1}^{N}\alpha_{t}(j)a_{ji}\end{bmatrix}b_i(o_{t+1}), \tag{5-22}
    \; i=1,2,\cdots,N
    $$

3. 终止

    + $P(O\mid \lambda) = \sum \limits_{i=1}^{N}\alpha_{T}{(i)} \tag{5-23}$

> 时间复杂度分析: 对于有N个状态数的状态集合,序列长度为T的HMM模型,时间复杂度为$O(TN^2)$

#### **3.3.2.2 后向算法**


**后向概率**:

给定隐马尔科夫模型$\lambda$,定义在时刻t状态为$q_i$的条件下,从$t+1$到$T$的部分观测序列为
$o_{t+1}, o_{t+1}, \cdots, o_{T}$的概率:

$$
\beta_{i}(t) = P(o_{t+1},o_{t+2},\cdots, o_T \mid i_t=q_i,\lambda)\tag{5-24}
$$

**后向算法的步骤**

**输入**: 隐马尔科夫模型$\lambda$,观测序列$O$;  
**输出**: 观测序列概率$P(O\mid \lambda)$

1. 初值

    + $\beta_{T}(i) = 1,\quad i=1,2,\cdots,N \tag{5-24}$

2. 递推

    + 对于$t=T-1, T-2, \cdots, 1$  
    $\beta_t(i) = \sum \limits_{j=1}^{N}a_{ij}b_j(o_{t+1})\beta_{t+1}(j), \quad i=1,2,\cdots,N \tag{5-25}$

3. 终止

    + $P(0\mid\lambda) = \sum \limits_{i=1}^{N}\pi_{i}b_{i}(o_1)\beta_1(i) \tag{5-26}$

实际上,利用前向概率和后向概率的定义可以将观测序列概率$P(O\mid \lambda )$统一写成:

$$
P(O \mid \lambda) = \sum \limits_{i=1}^{N} \sum \limits_{j=1}^{N}
\alpha_{t}(i)a_{ij}b_{j}(o_{t+1})\beta_{t+1}(j), \quad t=1,2,\cdots, T-1
\tag{5-27}
$$

&ensp; 当$t=1$时对应到式(5-22), $t=T-1$对应到式(5-27)


**根据前向概率和后向概率,我们可以得到一些对后面学习模型有用的概率与期望值**

1. 给定$\lambda$与观测$O$,在时刻t处于状态$q_i$的概率

$$
\begin{aligned}
\gamma_{t}(i) &= P(i_t = q_t \mid O,\lambda) \\
 &= \frac{P(i_t = q_i, O \mid \lambda)}{P(O \mid \lambda)}
 \xrightarrow{\alpha_t(i)\beta_t(i) = P(i_t=q_i,O \mid \lambda)}
 \frac{\alpha_t(i)\beta_t(i)}{\sum \limits_{j=1}^{N}\alpha_t(j)\beta_t(j)}
\end{aligned} \tag{5-28}
$$

2. 给定模型$\lambda$和观测$O$.在时刻t处于状态$q_i$且在时刻$t+1$处于状态$q_j$的概率

$$
\begin{aligned}
\epsilon_t(i,j) &= P(i_t=q_i, i_{t+1}=q_j \mid O,\lambda) \\  
&= \frac{P(i_t=q_i, i+{t+1}=q_j, O \mid \lambda)}{P(O\mid \lambda} \\
&= \frac{P(i_t=q_i, i_{t+1}=q_j, O\mid \lambda)}
{\sum \limits_{i=1}^{N} \sum \limits_{j=1}^N P(i_t=q_i,i_{t+1}=q_j,O\mid \lambda)}
\end{aligned}\tag{5-29}           \\
又P(i_t=q_i, i_{t+1}=q_j,O\mid \lambda) = \alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j) \\
\epsilon_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}
{\sum \limits_{i=1}^{N} \sum \limits_{j=1}^N(\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j))}
$$


## **3.4. 学习算法**

&ensp;下面我们来看看HMM的学习算法,根据训练数据是包括观测序列和对应的状态序列还是只有观测序列,可以分别由
监督学习与非监督学习实现.下面首先介绍监督学习再介绍非监督学习算法-**Baum-Welch(也是EM算法)**

### **3.4.1 监督学习方法**

&ensp; 假设已给训练数据包含$S$个长度相同的观测序列和对应的状态序列$$\{(O_1,I_1),(O_2,I_2),\cdots,(O_s,I_s)\}$$
,那么很容易可以想到用 **极大似然估计法(可以简单的用频率替代概率)** 来估计HMM模型的参数

1. 转移概率$a_{ij}$的估计

    + 设样本中时刻$t$处于状态$i$,时刻$t+1$转移到状态$j$的**频数**为$A_{ij}$,那么状态转移概率$a_{ij}$的估计是:

    $$
    \hat{a_{ij}} = \frac{A_{ij}}{\sum \limits_{j=1}^{N}A_{ij},\quad i=1,2,\cdots,N;\;j=1,2,\cdots,N}\tag{5-30}
    $$

2. 观测概率$b_j(k)$的估计

    + 设样本中状态为$j$并观测为$k$的频数是$B_{jk}$,那么状态为$j$观测为$k$的概率$b_j(k)$的估计是:

    $$
    \hat{b_j}(k) = \frac{B_{jk}}{\sum \limits_{k=1}^{M}B_{jk}},\quad j=1,2,\cdots,N;\;k=1,2,\cdots,M \tag{5-31}
    $$

3. 初始状态概率$\pi_i$的估计$\hat{\pi_i}$为$S$个样本中初始状态为$i$的频率



### **3.4.2 Baum-Welch算法**

上面监督学习需要使用训练数据,而人工标训练数据往往代价很高,这就要用到非监督学习算法

&ensp;假设给定训练数据只包含$S$个长度为$T$的观测序列${0_1, O_2, \cdots,O_s}$而没有对应的状态序列,目标是学习
隐马尔科夫模型$\lambda=(A,B,\pi)$的参数.HMM模型实际上是一个包含由隐变量的概率模型:

$$
P(O\mid \lambda) = \sum \limits_{I}P(O\mid I,\lambda)P(I\mid \lambda) \tag{5-32}
$$

那么如何学习这个模型的参数呢? 这就需要我们利用最开始讲的EM算法了.

1. 确定完全数据的对数似然函数

    + 所有观测数据写成$O=(o_1,o_2,\cdots,o_T)$,所有隐数据写成$I=(i_1,i_2,\cdots,i_T)$,
    完全数据是$(O,I)=(o_1,o_2,\cdots,o_T,i_1,i_2,\cdots,i_T)$.完全数据对数似然函数为:

    $$
    P(O,I\mid \lambda)
    $$

2. EM算法的E步

    + 求$Q$函数$Q(\lambda,\bar{\lambda})$

    $$
    \begin{aligned}
    Q(\lambda,\bar{\lambda}) &= E_t[\log P(O,I\mid \lambda) \mid O,\bar{\lambda}] \\
    &= \sum \limits_{I}\log P(I\mid O,\bar{\lambda})\log P(I,O\mid \lambda) \\
    &= \sum \limits_{I} \log \frac{P(I,O\mid \bar{\lambda})}{P(O\mid \bar{\lambda})}
        \log P(I,O \mid \lambda) \\
    & \xrightarrow{省去对\lambda而言的常数因子\frac{1}{P(O\mid \bar{\lambda})}}
    \sum \limits_{I}\log P(I,O\mid \bar{\lambda})P(I,O\mid \lambda)
    \end{aligned} \tag{5-33}
    $$
    其中,$\bar{\lambda}$是隐HMM参数的当前估计值,$\lambda$是要极大化的HMM参数.又  

    $$
    P(O,I \mid \lambda) = \pi_{i_1}b_{i_1}(o_1)a_{i_1 i_2}b_{i_2}(o_2)\cdots a_{i_{T-1}\; i_T}b_{i_T}(o_T)
    $$
    于是函数$Q=(\lambda, \bar{\lambda})$可以写成:

    $$
    Q(\lambda,\bar{\lambda}) = \sum \limits_{I}\log \pi_{i_1}P(O,I\mid \bar{\lambda})
        + \sum \limits_{I}\left(\sum \limits_{t=1}^{T-1} \log a_{i_1 i_t+1}\right)P(O,I \mid \bar{\lambda})
        + \sum \limits_{I}\left(\sum \limits_{t=1}^{T} \log b_{i_t}(o_t)\right)P(O,I\mid \bar{\lambda})  \tag{5-34}
    $$

2. EM算法的M步

    + 极大化$Q$函数$Q(\lambda, \bar{\lambda})$求模型参数$A,B,\pi$

    由于到极大化的参数在式5-34中单独出现在3个项中,所有只需对各项**分别极大化**.
    在极大化的过程中,分别利用$\sum \limits_{i=1}^{N}\pi_i = 1,\; \sum \limits_{j=1}^{N}a_{ij}=1,\;\sum \limits_{k=1}^{M}b_j(k)=1$
    这三个约束条件,再用**拉格朗日乘子法**,对待求参数求偏导并令结果等于0,即可解出.
    由于求导过程比较繁琐,大家如果感兴趣可以参考[5].我这里直接得出结论:

    $$
    \begin{aligned}

    a_{ij} &= \frac{\sum \limits_{t=1}^{T-1}\epsilon_{t}(i,j)}{\sum \limits_{t=1}^{T-1}}\gamma_t(i) \\
    b_j(k) &= \frac{\sum \limits_{t=1,o_t=v_k}^{T}\gamma_t(j)}{\sum \limits_{t=1}\gamma_t(j)} \\
    \pi_i  &= \gamma_1(i)    
    \end{aligned}

    $$

## **3.5. 预测算法**

HMM有两种预测算法: 近似算法与维特比算法.
由于本章篇幅太长了，且这两个算法都比较简单，这里就不介绍.


**参考文献**

<span id = "id1">[1] 李航, 统计学习方法. </span>  
[2] A. P. Dempster; N. M. Laird; D. B. Rubin. Maximum Likelihood from Incomplete Data via the EM Algorithm.  
[3] Hidden Markov model: https://en.wikipedia.org/wiki/Hidden_Markov_model.  
[4] Markov_chain: https://en.wikipedia.org/wiki/Markov_chain.  
[5] Daniel Ramage,Hidden Markov Models Fundamentals, CS229 Section Notes.  
