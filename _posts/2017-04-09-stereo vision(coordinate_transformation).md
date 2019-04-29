---
layout: post
title: (1)坐标转化
date: 2017-04-09 16:04
description: 相机模型中的三个坐标转化
tag: stereo vision
---

# **前言**
* 立体匹配和SFM中最基本的问题就是坐标转化问题，因此在讲stereo vision之前，很有必要先介绍一下坐标转化问题。  

# **(第一章) 坐标转化**

## **1. 摄像机模型**

&ensp;   在讲坐标转化之前，很有必要介绍一下摄像机模型。摄像机模型中，最简单的应该就是针孔模型。在该模型中，我们不妨想象光线是从场景中的某个物体发射来，且仅仅是来自某个Q点的光线。最后，该点将在图像平面（projective plane )上被聚焦。如图1-1：

![图1-1](/images/posts/stereo vision/original_model.PNG){:weight="450px" height="239px"}


 上图中，由简单的相似三角形可得出:  

 $$
 -h =  f\frac{H}{Z} \tag{1}
 $$

 &ensp; 其中f是摄像机焦距。Z是摄像机到物体的距离，H是物体的高度（长度），h是图像平面上物体的高度（长度）。处于处理上简单的目的，我们调整针孔平面和图像平面的位置，并将针孔平面只留下一个针孔（作为投影中心），这也是实际的摄像机模型。如图1-2：    

 ![图1-2](/images/posts/stereo vision/fix_model.PNG){:weight="504px" height="180px"}


&ensp; 上图中，光轴与图像平面的交点称为主点（principal point )。图1-1与1-2两种情况在数学上等价，即物体在图像平面上投影大小是相同的。这样（1）式中即变为：$ \frac{h}{f} = \frac{X}{Z} $ 。此时物体的像是正立的，因此负号去掉了。  

### 主点位置

&ensp; 上面提到的主点，在理论上应该会在图像平面的中心，但是实际工艺的限制，通常会有一定的偏移。由此，引入实际主点坐标g( cx , cy )，表示实际的主点位置：例如一个640*480的图像，其主点一般不是（320，240），而可能会是（312，250）、（341，230）等。  


## **2. 坐标转化**

### 2.1 世界坐标系与摄像机坐标系

![图1-3](/images/posts/stereo vision/translate_coor.PNG){:weight="516px" height="388px"}


如图1-3 ，假设$(O_w,X_w,Y_w,Z_w)$为世界坐标系，$(O_c,X_c,Y_c,Z_c)$为摄像机坐标系。则可以用一个旋转矩阵 R 和一个平移矩阵 t 来表示这两个坐标系之间的变换。R 为3X3的正交变换矩阵，t 为3X1的向量。用齐次坐标系表示为：

$$
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix} = \begin{bmatrix}
R & T \\
0 & 1 \\
\end{bmatrix} \begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix} \tag{2}
$$

&ensp; 在所有的三维重建恢复都是为了求这两个参数而做的，这些都是后话了。  

### 2.2 摄像机坐标系与图像平面坐标系和像素坐标系
&ensp; 之所以将这三个坐标系合在一起讲，是因为它们之间的转化是一气呵成的，便于理解。在图1-3中，点Q在摄像机坐标系下假设坐标为$(x_c,y_c,z_c)$，其在图像平面上的投影点的坐标为$(x_u,y_u)$。则两者之间的变换关系为：  

$$
\left\{
\begin{aligned}
  x_u = \frac{f}{z_c}x_c  \\
  y_u = \frac{f}{z_c}y_c
 \end{aligned}
 \right.
 \tag{3}
$$

&ensp; 前面提到的主点还记得吗，图像坐标系就是以主点g为坐标原点，而像素坐标系的原点O就是一个图像位置为（0，0）的位置。这两个坐标系之间的转换一般就只有平移变化了，如下图：  

![图1-4](/images/posts/stereo vision/image_coor.PNG){:weight="450px" height="289px"}


&ensp;这里的g坐标即是上面摄像机模型谈到的（cx, cy)。这样转化关系可写为：  

$$
\left\{
  \begin{aligned}
  u = \frac{x_u}{dx}+c_x \\
  v = \frac{y_u}{dx}+cy
  \end{aligned}
  \right.
  \tag{4}
$$

&ensp;
其中dx 与 dy 分别表示单位像素在水平和垂直方向上的物理长度，即单位为：毫米/每像素。在实际上这两个值一般不相等，因为每个像素点在廉价的摄像机上是矩形而不是正方形。可以看到，若最终得到的坐标关系中留下dx、dy，那么我们必须知道摄像机出产时的具体参数，这对以后的标定来说无疑是巨大障碍。但是，我们可以这样联立公式（2），摄像机坐标系到像素坐标系转化即为：

$$
\left\{
  \begin{aligned}
  u = \frac{f/dx}{z_c}x_c + c_x \\
  v = \frac{f/dy}{z_c}y_c + cy
  \end{aligned}
  \right.
  \tag{5}
$$

令 $\frac{f}{dx} = fx$ 、$\frac{f}{dy} = fy$， 这里f的单位是毫米，分别与dx，dy相除恰好单位为像素，这样就可以不考虑物理细节了。则转化关系就和《学习OpenCv 中文版》中的转化关系类似了。  
&ensp; 最后联立世界坐标系和相机坐标系的转化关系，用齐次坐标表示即为：

$$
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix} = \frac{1}{z_c}\begin{bmatrix}
f_x & 0 & u_0 & 0 \\
0 & f_y & v_0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix} \begin{bmatrix}
x_w \\
y_w \\
z_w \\
1
\end{bmatrix}
\tag{6}
$$

&ensp;上面（6）中令:  

$$
K = \begin{bmatrix}
f_x & 0 & u_0 & 0 \\
0 & f_y & v_0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\tag{7}
$$

<br>
&ensp; 其中 **K** 就是熟知的内参矩阵了。至于为什么最后一列要加上，仅仅只是为了方便矩阵运算。 $\left[ \begin{array}{c|c} R & t \end{array}\right]$ 称为外参矩阵，关系为：

$$
\begin{bmatrix}
R|t
\end{bmatrix} = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\tag{8}
$$

<br>
&ensp; 在接下来的章节中，就是围绕如何求得 K  矩阵，以及如何应用一些方法从 **K** 矩阵得到 $\left[ \begin{array}{c|c} R & t\end{array} \right]$ 矩阵。  

<p style="color:red;font-size:1.em;">参考资料</p>
<ol>
  <li>《learning opencv》</li>
</ol>
