---
layout: post
title: (2)Using auto
date: 2017-05-08 21:02
description: 本章举了一些利用auto可以避免掉进c++典型的坑里
tag: modern c++
---

## **1. 为什么使用auto**

&ensp; 在c++11之前，声明一个变量的时候如果忘记初始化，编译器并不会报错，这可能导致后面不小心用到了未初始化的变量值。
在c++11后，用auto编译器会对没有初始化的变量报错，这就避免了上面的问题。

当然auto关键字引入之后通常会有如下作用：

- 程序变得简洁，避免冗长的显式定义类型

- 能够避免程序员因为对类型的误配操作导致运行结果错误或者效率低下

书上举了个例子：

~~~
std::vector<int> v;

...

unsigned sz = v.size(); //这个语句在32-bit windows下没有问题，但是在64-bit Windows下会有问题
                        //因为v.size()类型为std::vector<int>::size_type，在32-bit Windows下
                        //unsigned与size_type都是32bit，然而在64-bit Windows下unsigned仍然为32bit，s
                        //std::vector<int>::size_type却是64bit

auto sz = v.size();     // auto可以自动推导出std::vector<int>::size_type类型    

~~~

## **2. auto有哪些坑**

- 不可见的代理类(proxy types)可能使auto在初始化表达式中推导出错误出我们不想要的类型

&ensp; 前面一章已经提过auto作为返回值类型可能出现的问题，这里再补充一点。auto处理隐式代理类
可能会有问题：

~~~

std::vector<bool> features;

...

bool f1 = features[5]; // 正确，将std::vector<bool>::reference类型的object转为bool类型（注意不能转为bool &类型)

auto f2 = features[5]; // 错误，auto并不等价与上式，再STL中std::vector<T>的[]操作符除了T为bool类型外，其他都
                       // 是返回T类型的引用。当T为bool时，由于c++禁止对bits的引用，故这里用到了std::vector<bool>::reference这个
                       // 代理类。至于f2此时具体是什么类型，比较复杂，只要记住结论就可以!
~~~

因此可以得出：当程序用auto时，避免诸如形式为```auto someVar = expression of "invisible" proxy class type```的不可预见结果的代码！

## **3. 如何发现"invisible" proxy class type**

对于如何及时发现"invisible" proxy class type，Scott Meyers并没有给出方便的方法，只是提到可以查使用的类型的文档或者当编译出错时跟踪等
比较麻烦的方法。这无疑增加了很多麻烦，我们只能尽量在写程序时用自己熟悉的STL模板类 ┭┮﹏┭┮

## **4. 如何解决auto问题**

&ensp; 对于代理类给auto带来的麻烦，我们可以通过显式初始化器```cast```：

~~~
auto f2 = static_cast<bool>(features[5]);  //使用static_cast<bool>，显式将std::vector<bool>::reference object转为bool

// 自定义类： 一个Matrix类的operator+可以返回一个代理类对象如Sum<Matrix,Matrix>，而不是Matrix对象本身，这样可以提高计算的效率
Matrix sum = m1 + m2 + m3 + m4

auto sum = static_cast<Matrix>(m1+m2+m3+m4); //当用了代理类对象后，可以用static_cast<Matrix>强制转换为Matrix

~~~
