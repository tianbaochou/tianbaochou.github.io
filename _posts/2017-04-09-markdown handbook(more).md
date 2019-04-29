---
layout: post
title: (2)markdown进阶
date: 2017-04-09 22:39
description: 这篇博客将记录markdown中表格生成和常用的Latex公式语法
tag: markdown
---

> 上一篇博客介绍了markdown基本的语法,本篇将介绍Latex公式语法,并且记录常用的符号代码
> 接着介绍一下表格的生成

## **1. Latex**

### 1. 公式标识
+ ``` $$ Latex公式 $$ ``` 另起一段开始公式
+ ``` $Latex公式$ ``` 嵌在当前行的公式

### 2. 基本符号

#### 2.1 希腊字母的表示

| 字母  |  代码 | 字母  | 代码 |
|:--------:|:--------:|--------|--------|
|A|``` A ```|$\alpha $|```\alpha ```|
|B|``` B ```|$\beta$|```\beta```|
|$\Gamma$|```\Gamma```|$\gamma$|```\gamma```|
|$\Delta $|```\Delta ```|$\delta $|```\delta ```|
|E|``` E ```|$\epsilon $|```\epsilon ```|
|Z|``` Z ```|$\zeta $|```\zeta ```|
|H|``` H ```|$\eta $|```\eta ```|
|$\Theta $|```\Theta ```|$\theta $|```\theta ```|
|I|```I ```|$\iota $|```\iota ```|
|K|```K ```|$\kappa $|```\kappa ```|
|$\Lambda $|```\Lambda ```|$\lambda $|```\lambda ```|
|M|```M ```|$\mu $|```\mu ```|
|N|```N ```|$\nu $|```\nu ```|
|$\Xi $|```\Xi ```|$\xi $|```\xi ```|
|O|```O ```|$\omicron $|```\omicron ```|
|$\Pi $|```\Pi ```|$\pi $|```\pi ```|
|P|```P ```|$\rho $|```\rho ```|
|$\Sigma $|```\Sigma ```|$\sigma $|```\sigma ```|
|T|```T ```|$\tau $|```\tau ```|
|$\Upsilon $|```\Upsilon ```|$\upsilon $|```\upsilon ```|
|$\Phi $|```\Phi ```|$\phi $|```\phi ```|
|X|```X ```|$\chi $|```\chi ```|
|$\Psi $|```\Psi ```|$\psi $|```\psi ```|
|$\Omega $|```\v ```|$\omega $|```\omega ```|

#### 2.2 常用数学符号

| 功能 | 代码 | 案例 | 案例代码 |
|:--------:|:--------:|--------|--------|
|```加 ```|```+ ```|$a+b $|```$x+y $```|
|```减 ```|```- ```|$a-b $|```$a-b $```|
|```叉乘 ```|```\times ```|$a \times b $|```$a \times b $```|
|```点乘 ```|```\cdot ```|$a \cdot b $|```$a \cdot b $```|
|```星乘 ```|```\ast ```|$a \ast b $|```$a \ast b $```|
|```除 ```|```\div ```|$a \div b $|```$a \div b $```|
|```分数 ```|```\frac ```|$\frac{a}{b} $|```$\frac{a}{b} $```|
|```上标 ```|```^ ```|$a^b $|```$a^b $```|
|```下标 ```|```_ ```|$a_b $|```$a_b $```|
|```开二次方 ```|```\sqrt ```|$\sqrt a $|```$\sqrt a $```|
|```开方 ```|```\sqrt ```|$\sqrt[a]{b} $|```$\sqrt[a]{b} $```|
|```加减 ```|```\pm ```|$a \pm b $|```$a \pm b $```|
|```减加 ```|```\mp ```|$a \mp b $|```$a \mp b $```|
|```等于 ```|```= ```|$ a = b $|```$ a = b$```|
|```小于等于 ```|```\leq ```|$a \leq b $|```$a \leq b $```|
|```大于等于 ```|```\geq ```|$a \geq b $|```$a \geq b $```|
|```不小于等于 ```|```\nleq ```|$ a \nleq b $|```$a \nleq b $```|
|```不大于等于 ```|```\ngeq ```|$ a \ngeq b $|```$a \ngeq b $```|
|```不等于 ```|```\neq ```|$a \neq b $|```$a \neq b $```|
|```约等于 ```|```\approx ```|$ a \approx b $|```$a \approx b$```|
|```恒等于 ```|```\equiv ```|$ a \equiv b $|```$a \equiv b $```|
|```累加 ```|```\sum ```|$ \sum_{i=a}^{b} $|```$\sum_{i=a}^{b} $```|
|```累乘 ```|```\prod ```|$\prod_{i=a}^{b} $|```$\prod_{i=a}^{b} $```|
|``` ```|```\bigcup ```|$\bigcup_{i=a}^{b} $|```$\bigcup_{i=a}^{b} $```|
|```积分 ```|```\int ```|$\int_{a}^{b}f(x)\text{ d}t $|```$\int_{a}^{b}\text{ d}t $```|
|```重积分 ```|```\iint ```|$\iint_{a}^{b}f(x,y){d}\sigma $|```$\int_{a}^{b}f(x,y){d}\sigma $```|

#### 2.3 常用的二元关系

| 功能 | 代码 | 案例 | 案例代码 |
|:--------:|:--------:|--------|--------|
|```远小于```|```\ll ```|$\ll $|```$ a \ll b $```|
|```远大于 ```|```\gg ```|$ \gg$|```$a \gg b $```|
|```偏序```|```\prec ```|$\prec $|```$ a \prec b$```|
|```偏序 ```|```\preceq ```|$ \preceq $|```$ a \preceq b$```|
|```真子集 ```|```\subset ```|$\subset $|```$ a \subset b$```|
|```真子集 ```|```\supset ```|$ \supset$|```$ a \subsubset b$```|
|```子集 ```|```\subseteq ```|$ \subseteq$|```$a \subseteq b$```|
|```相似 ```|```\sim ```|$ \sim$|```$A \sim B $```|
|```属于```|```\in ```|$ \in$|```$a \in b $```|
|```不属于```|```\notin```|$\notin $|```$a \notin b $```|
|``` ```|```\simeq ```|$ \simeq$|```$ a \simeq b$```|




### 3. 矩阵和大括号

#### 3.1. 矩阵

+ 矩阵使用 ```$$ \begin{matrix} ... \end{matrx} ```生成，且矩阵中每行用``` \\\ ```结束，
元素之间用&来分隔

##### 3.1.1.  不带括号的矩阵

~~~
$$
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{matrix}
\tag{3-1-1}
$$
~~~

**效果如下**

$$
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{matrix}
\tag{3-1-1}
$$

##### 3.1.2.  带括号的矩阵{...}

~~~
$$
\begin{Bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{Bmatrix}
\tag{3-1-2}
$$
~~~

**效果如下**

$$
\begin{Bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{Bmatrix}
\tag{3-1-2}
$$

##### 3.1.3.  带括号的矩阵[...]

~~~
$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\tag{3-1-3}
$$
~~~

**效果如下**

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\tag{3-1-3}
$$

##### 3.1.4.  带括号的矩阵(...)

~~~
$$
\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix}
\tag{3-1-4}
$$
~~~

**效果如下**

$$
\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix}
\tag{3-1-4}
$$

##### 3.1.5.  带括号的矩阵||...||

~~~
$$
\begin{Vmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{Vmatrix}
\tag{3-1-5}
$$
~~~

**效果如下**

$$
\begin{Vmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{Vmatrix}
\tag{3-1-5}
$$

##### 3.1.6.  带括号的矩阵[...]

~~~
$$
\begin{vmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{vmatrix}
\tag{3-1-6}
$$
~~~

**效果如下**

$$
\begin{vmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{vmatrix}
\tag{3-1-6}
$$

##### 3.1.7.  带省略号的矩阵

一般使用 ```\cdots```作为横省略点；```\vdots```作为竖省略点；```\ddots```作为斜省略点:  

~~~
$$
\left[
\begin{matrix}
1 & 2 & \cdots & 4 \\
4 & 5 & \cdots & 6 \\
\vdots & \vdots \ ddots & \vdots \\
8 & 9 & \cdots & 0 \\
\end{matrix}
\right]
\tag{3-1-7}
$$
~~~

**效果如下**

$$
\left[
\begin{matrix}
1 & 2 & \cdots & 4 \\
4 & 5 & \cdots & 6 \\
\vdots & \vdots & \ddots & \vdots \\
8 & 9 & \cdots & 0 \\
\end{matrix}
\right]
\tag{3-1-7}
$$

##### 3.1.8.  增广矩阵

可以用```array```配合```\left[ ... \right]```来构造

~~~
$$
\left[
\begin{array}{cc|c}
1 & 2 & 3 \\
4 & 5 & 6
\end{array}
\right]
\tag{3-1-8}
$$
~~~

**效果如下**

$$
\left[
\begin{array}{cc|c}
1 & 2 & 3 \\
4 & 5 & 6
\end{array}
\right]
\tag{3-1-8}
$$

##### 3.1.9.  行间矩阵

可以用```\bigl(\begin{smallmatrix} ... \end{smallmatrix}\bigr)```来将矩阵缩小的行间  

```
$\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix} \bigr)$
```

**效果如下**

使用矩阵$\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix} \bigr)$来...

### 3.2. 大括号

配合```\left\{ \begin... \end{aligned}\right.```
注意最后的点

~~~
$$ f(x)=\left\{
\begin{aligned}
x & = & \cos(t) \\
y & = & \sin(t) \\
z & = & \frac{x}{y}
\end{aligned}
\right.
\tag{3-2}
$$
~~~

**效果如下**

$$ f(x)=\left\{
\begin{aligned}
x & = & \cos(t) \\
y & = & \sin(t) \\
z & = & \frac{x}{y}
\end{aligned}
\right.
\tag{3-2}
$$

### 3.3.  等号对齐

~~~

$$
\begin{align*}
x & = (x+a)^2 \\
 & =  x^2 + a ^2 + 2xa
\end{align*}
\tag{3-3}
$$

~~~
效果:

$$
\begin{align*}
x & = (x+a)^2 \\
 & =  x^2 + a ^2 + 2xa
\end{align*}
\tag{3-3}
$$

上面使用align对齐.若要使用gather，则对齐方式按照全局方式对齐:

~~~
$$
\begin{gather*}
E(x) = \lambda \qquad D(x) = \lambda \\
E(\bar{x}) = \lambda \\
D(\bar{x}) = \frac{\lambda}{n} \\
E(S^2) = \frac{n-1}{n}\lambda \\
\end{gather*}
$$
~~~

效果:

$$
\begin{gather*}
E(x) = \lambda \qquad D(x) = \lambda \\
E(\bar{x}) = \lambda \\
D(\bar{x}) = \frac{\lambda}{n} \\
E(S^2) = \frac{n-1}{n}\lambda \\
\end{gather*}
$$


>全局对齐方式可以采用在documentclass或者amsmath包前面加上参数fleqn: ``` \documentclass{fleqn}{article} ``` 设置
>为左对齐，其默认是居中对齐

## **2. 表格生成**

由于平常用的表格一般比较简单（复杂的不建议用markdown语法生成，太复杂了）
因此markdown在这方面也比较简单，用上面生成的希腊字母的表，截取一段即可掌握表格生成

~~~
| 字母  |  代码 | 字母  | 代码 |
|:--------:|:--------:|--------|--------|
|A|``` A ```|$\alpha $|```\alpha ```|
|B|``` B ```|$\beta$|```\beta```|
~~~

**效果如下**

| 字母  |  代码 | 字母  | 代码 |
|:--------:|:--------:|--------|--------|
|A|``` A ```|$\alpha $|```\alpha ```|
|B|``` B ```|$\beta$|```\beta```|

&ensp; 其中```|:----|```代表左对齐；```|:-------:|```代表居中；```|-------:|```代表右对齐,
```|...|```表项内可以使用Latex公式等,也可以嵌套；```-```的个数依据每个表项的相对长度适当增减

## Atom编辑器markdown中的快捷键

$ensp; 今天发现Atom编辑器中有一些很有用的markdown插入link、table、img、code等快捷键，整理一些如下：

| 功能描述     | 快捷键 |
| :------------- | :------------- |
| 插入图像(本地)       | ```img + tab```       |
| 插入图像(引用)       | ```rimg + tab```       |
| 插入表格       | ```table + tab```       |
| 插入引用       | ```ref + tab```       |
| 引用链接       | ```rl + tab```       |
| Copyright       | ```legal + tab```       |
| 插入代码       | ```code + tab```       |
| 字体加粗       | ```b + tab ```       |
| 字体斜体       | ```i + tab```       |
| Todo       | ```t + tab```       |

&ensp; Atom允许你自定义快捷键，打开```File->Snippets```，则可以添加增加yaml头的快捷键：

~~~

'.source.gfm':
  'yaml':
    'prefix': 'yaml'
    'body': """
      ---
      layout: $1
      title: $2
      date: $3
      description: $4
      tags: $5
      ---

      $7
    """
  'code':
    'prefix': 'code'
    'body': """
      ~~~
      $1
      ~~~
    """
~~~
上面也改了code的插入的默认符号，因为之前那个符号在github pages中支持不好!!
