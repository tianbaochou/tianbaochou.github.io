---
layout: post
title: (1)基本概念
date: 2017-04-24 23:13
description: 主要介绍convex的一些基本概念
tag: convex optimization
---

# **前言**

 > <Convex Optimization>这本书是知乎上的ML大神们普遍推荐的书本，本着大概率情况下，应该对自己也会帮助较大，于是便从网上下载了这本英文原版书，下载后大概看了一下，全书分为三个部分:  

> + theory 主要讲Convex Optimization的基础知识,特别涉及到很多代数分析
> + application 讲述了Convex Optimization比较重要的应用
> + algorithm 讲了许多优化方法，例如：梯度下降、牛顿法等

&ensp; 整本书正文大概600多页, 刚看的时候着实欣喜，心想几百页应该能比较快解决，然后又看了看Introduction部分,发现围绕着$ minimize \quad f_0(x) \quad  f_i(x)\leq b_i,  \quad i = 1,...,m $展开,本科期间也大概了解过,这就大大增加了我的信心,然后就一头扎进去了, 看了几天,发现Theory中Convex Set部分就差点让我吐血。各种概念，各种代数分析的外延知识，很多点Stephen Boyd提了一下没有解释，然后我就云里雾里，压根不知道那个结论哪里来的，哭死。
<p style="color:red;font-size:3.em;front-weight:bold">&ensp;&ensp;基于此，写这个系列的博客更新进度肯定很慢，而且难免有纰漏或者理解不对，要是有童鞋能看出，不妨指出一起讨论，互相学习。本科不是数学班科出生，考研又浪费了很多大好时光补数学知识（汗颜!）因此希望这个专题可以一边写学习笔记，一遍重新理清一些关系(好羡慕领悟力高的娃!)，期间将会和《Algebra analysis》一起同步学习。
</p>


# **基本概念**

## 1. What means Convex in chinese  

&ensp; 我想大部分童鞋看到Convex的图像内心是崩溃的，这和本科高等数学中的凸的定义正好相反啊! 还好之前好像看到一个视频说台湾凹凸的定义和大陆这边也正好相反，估计是定义不同，特地查了资料在此解释一下:  

&ensp; 数学上的凸函数和直观上是相反的： 凸函数 $ \Leftrightarrow $函数epigraph是一个凸集，以大家最熟悉的二次函数为例:   

![图1-1](/images/posts/convex optimization/chapter one/chapter_one_1.PNG)

&ensp; 上图中阴影部分为函数f(x)的epigraph,它是一个convex set:
$$
epi f = \{(x,t)|x\in{domf},f(x) \leq t\}
$$
**即若f(x)为凸则其epigraph为凸集**

## 2.凸优化和线性规划的区别

&ensp; 先来看看优化问题的一般形式:

$$

minimize \quad f_0(x) \\
subject to \quad f_i(x) \leq b_i \quad ,i=1,2,...,m \tag{1}

$$

&ensp; 其中:  

+ $ f_0, \  f_1, \ , ...,\ ,f_m $为线性函数，即: $ f_i( \alpha x + \beta y ) = \alpha f_i(x) + \beta f_i(y) $
则以上形式为线性规划问题。  
+ $ f_0, \ f_1, \ ,..., \ , f_m $为凸函数，即：$ f_i(\alpha x + \beta y ) \leq \alpha f_i(x) + \beta f_i(y) $
则以上形式为凸优化问题。
