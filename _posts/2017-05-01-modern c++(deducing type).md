---
layout: post
title: (1)Deducing Type
date: 2017-05-01 23:43
description: 本章主要介绍了类型推导相关知识，比较繁琐，适合当工具查(Remember Some Important Thing!!!)
tag: modern c++
---

**前沿**

> C++和C是很多人，包括本人的程序入门语言，从大一开始，到工作，足足学习了5年左右。虽说有效学习时间远没有那么多，
但是相比于其他语言(或者压根没学过其他语言)(●ˇ∀ˇ●)来说已经很多了，不过说实话，相对于Python和Java，C++
确实太繁杂了。比如说，你考个研，放它好几个月，当你重拾起的时候，基本只剩下，一些基本语法，多态这些记得，像模板推导，
一些繁琐的坑，忘得差不多了（记性好的童鞋别鄙视哈(。・∀・)ノ)。好后悔当初不改入Java坑(ノへ￣、)

> 基于此，还是写个专题，将现代C++的许多特性和坑做分类，以便后面工作学习可以方便查阅。

> **本专题实际可以认为是《Effective Modern C++》的学习笔记，只不过会加入一些，个人比较容易混肴的其他知识！** :smile:


## **1. Template type deduction**

 **lvalue and rvalue**

 &ensp; 简单点来说，左值和右值的区别就是左值能够知道它的存储地址，右值无法知道它的存储地址,
 例如 ```int x = 27 ```,这里27就是右值，x就是左值。因为我们不知道27会被存储在那个地址.

假设我们有如下形式的模板定义与调用：

~~~

template<typename T>
void f(ParamType param);
...
f(expr); // 将会从expr推导T和ParamType

~~~

&ensp; 下面将从根据ParamType的形式分为三种情况

### **1.1 ParamType是一个引用或者指针，但是非Universal Reference**

推导步骤如下：

1. expr若为引用，则忽略引用
2. 根据1结果从expr中推到ParamType，并推出T

~~~
// 不带const
template<typename T>
void f1(T& param);

int x1 = 27;
const int x2 = x;
const int x3 = x;

f1(x1); // ParamType: int&, T: int
f1(x2); // ParamType: const int&, T: const int
f1(x3); // ParamType: const int&, T: const int

// 带const
template<typename T>
void f2(const T&param);

int y1 = 27;
const int y2 = x;
const int & y3 = x;
f2(y1); //ParamType: const int&, T: int
f2(y2); //ParamType: const int&, T: int
f3(y3); //ParamType: const int&, T: int

// 指针
template<typename T>
void f3(T*param);

int z1 = 27;
const int *z2 = &x;

f3(&z1); //ParamType: int*, T: int
f3(z2);  //ParamType: int*, T: int
~~~


### **1.2 ParamType为Universal Reference**

推导步骤如下:

1. 若expr为lvalue，T和ParamType均被推到为lvalue Reference，这也是T被推到为引用类型的唯一情况
2. 若expr为rvalue, 则可以参照1.1

~~~
template<typename T>
void f(T&& param);

int x1 = 27;
const int x2 = x;
const int &x3 = x;

f(x1); //ParamType: int &, T: int &
f(x2); //ParamType: const int&, T: const int&
f(x3); //ParamType: const int&, T: const int&
f(27); //ParamType: int&&, T: int

~~~

***Remember: 只有Universal Reference在类型推导时会区分lvalue和rvalue参数***

### **1.3 ParamType既不是指针也不是引用**

&ensp;这说明参数传递是值传递，那么可以想象任何传递的参数，其本身数据附加特征,eg：const、引用、volatile都被忽略
因为传递的参数修饰不影响param本身

~~~
template<typename T>
void f(T param);

int x1 = 27;
const int x2 = x;
const int &x2 = x;

f(x1); //ParamType: int, T: int
f(x2); //ParamType: int, T: int
f(x3); //ParamType: int, T: int
~~~

### **1.4数组作为参数**

&ensp;当数组作为参数传递给模板函数时，若模板函数的参数为值传递，则数组退化为指针；
若模板函数的参数为引用，则模板推导可以将T推导为数组!!

~~~
const char name[] = "zbabby";

template<typename T>
void f1(T param){
  cout <<" size Reference :" << sizeof(T) << endl; //输出指针占用的字节 = 4
}

f(name); //name 为数组，但是这里退化为const char *

template<typename T>
void f2(T &param){
  cout << "size Reference : " << sizeof(T) << endl; //输出数组占用的直接 = 7
}

f2(name); //T推导为数组，即T被推导为const char[7],ParamType: const char (&)[7];

~~~

利用上面引用可以保留数组的特点，可以用模板函数推导数组元素个数：

~~~
template<typename T, std::size_t N>
constexpr std::size_t arraySize(T(&)[N]) noexcept
{
  return N;
}

int marray[] = {1,2,3,4,5};
cout << "size = " << arraySize(marray) << endl; // 输出5
~~~

### **1.5函数作为参数**

&ensp; 同数组一样，函数作为参数时也会退化为函数指针：

~~~
void SomeFunc(int, double);

template<typename T>
void f1(T param);  // param passed by value

template<typename T>
void f2(T& param); // param passed by ref

f1(SomeFunc);  // param 推导为函数指针: void(*)(int, double)

f2(SomeFunc);  // param 推导为函数引用: void(&)(int, double)
~~~

**总结**

- 在模板推导中传递的参数有引用视为无引用

- Universal Reference为徐需推导的类型时lvalue需特殊处理

- 推导类型为传值方式时，传递的参数带有const、volatile这些修饰无效

## **2. Auto type deduction**

在编译器推导auto具体类型时如同处理模板推导T一样，除了在uniform initialization时需要特殊
处理，一般可以参照上面的推导：

~~~
auto x1 = 27;        // x1 neither ptr nor reference
const auto x2 = x;   // x2 neither ptr nor reference
const auto &x3 = x;  // x3 为non-universal ref

auto&& uref1 = x1;    // x1 为int 且为lvalue，则uref1 : int &
auto&& uref2 = x2；   // x2 为const int 且为lvalue，则uref2: const int &
auro&& uref3 = 27;    // x3 为rvalue，则uref3: int&&

const char name[] = "zbabby";
auto arr1 = name;  // arr1: const char *
auto& arr2 = name; // arr2: const char(&)[7]

void someFunc(int, double);
auto fun1 = someFunc;   // fun1 : void(*)(int, double)
auto &fun2 = someFunc;  // fun2 : vodi(&)(int, double)

~~~

***例外***

当C++11支持uniform initialization后：

~~~
int x1 = 27;
int x1(27);
int x1 = {27};
int x2{27};
~~~

&ensp;上面四个语法都是支持的，即初始化x1为int，值为27。

&ensp;但是当auto用后面两种方式初始化变量时，其实际是声明一个```std::initializer_list<int>```类型，并且包含一个元素27！

~~~
auto x = {27} //相当于: std::initializer_list<int> x = {27};
auto x2 = {1,2,3.0}; //编译失败，因为列表中的元素类型要一致!
~~~

上面能自动推导是auto特有的规则，也就是：

~~~
auto x = {11,22,33};  // success

template<typename T>
void fun1(T param);

fun1({11,22,33});  // 失败，因为template type deduction不能推导出T的类型为std::initializer_list<int>，只能显式声明

template<typename T>
void fun2(std::initializer_list<T> init_list); //显式声明
fun2({11,22,33}); // T: int

~~~

下面比较苦逼的事情出现了:joy:，template type deduction 无法和auto一样自动推导```std::initializer_list<type>```类型，
然而在c++11中和c++14中auto 作为函数返回类型或者lambda参数类型时，却用的是template type 推导方式：

~~~

auto CreateList(){
  return {1,2,3}; //失败
}

std::vector<int> v;

auto resetV = [&v](const & newValue){ v = newValue;}  // C++ 14
...
resetV({1,2,3}); //失败
~~~

**总结**

- auto type deduction通常和template type deduction推导规则一样，唯一的区别是花括号```{}```初始化时，auto可以自动推导为```std::initializer_list<type>```类型

- 当auto用在函数返回值或者lambda函数参数时，用的时template type deduction规则!


## **3. Understand decltype**

c++11以后允许使用decltype作为来自动推导返回值，decltype使用案例如下：

~~~
template<typename Container, typename Index>
auto authAndAccess(Container & c, Index i) ->decltype(c[i]) // c++11
{
  ...

  return c[i];  //原样返回c[i]，eg Container 为vector<int>，则c[i]，返回int &；若
}

template<typename Container, typename Index>
decltype(auto) authAndAccess(Container & c, Index i) // c++14
{
  ...

  return c[i];  //原样返回c[i]，eg Container 为vector<int>，则c[i]，返回int &；若
}

std::deque<int> d1;
authAndAcces(d1,5) = 10;
~~~

&ensp;上面若将一个rvalue容器传给```authAndAccess```，则会出现问题，因为rvalue容器只是临时对象，将会在调用authAndAccess后
被销毁，这就会导致函数返回的值可能会是垃圾值。但若用户想将临时容器的元素复制到别处，例如下面场景：

~~~
std::deque<std::string> makeStringDeque();  // factory function

// make copy of 5th element of deque returned from makeStringDeque
auto s = authAndAccess(makeStringDeque(), 5);
~~~

这就不得不解决上面传递临时容器的问题。当然容易想到的是可以重载函数，分别处理lvalue和rvalue，但是这就需要维护两个函数，比较麻烦；
还有一种就是使用 **Universal reference**，```authAndAccess```函数可以声明如下：

~~~
template<typename Container, typename Index>
decltype(auto) authAndAccess(Container &&c, Index i ) //c 为一个Universal reference
{
  ...
  return std::forward<Container>(c)[i]; // 这里用std::forward，原因后面会讲
}
~~~

&ensp; 我们对不管是传递的Container是lvalue还是rvalue，都能够正确处理

关于decltype还有需要注意一点就是：

~~~
decltype(auto) f1(){
  int x = 0;
  return x; //decltype(x) : int
}

decltype(auto) f2(){
  int x = 0;
  return (x);  //decltype((x)): int，故f2返回int &
}

//上面两个函数是由于decltype在处理x、(x)是意义不同，x为T类型变量名，(x)为T类型lvalue expression，decltype在处理lvalue expression时会指示为T&
//这样，第二个函数返回了一个临时变量的引用，就有可能用了垃圾值
~~~

## **4. Know how to view deduced types**

&ensp; C++分别提供三种方式显示当前推导出的类型：

- IDE (IDE提示)
- Compiler Diagnostics(编译器诊断)
- Runtime Output(运行时输出)

**IDE**

这个最简单，例如：

~~~
const int x = 43;
//当鼠标放在y、z上面时，智能点的IDE都会提示出y为int 类型,z为const int *类型
auto y = x;
auto z = &x;
~~~

**Compiler Diagnostics**

~~~
const int x = 43;
auto y = x;

template<typename T>
class TD;    // TD == 'Type Displayer'即借助为定义的类模板，让编译器报错

TD<decltype(x)> xType;  // 提示错误包括x,y的类型
TD<decltype(y)> yType;
~~~

**Runtime output**

&ensp;使用typeid来输出类型，但是这个函数在处理复杂的类型时可能输出的类型是错误的！
例如：

~~~
template<typename T>
void f(const T &param){
  using std::cout;
  cout << "T = " << typeid(T).name() << endl;         
  cout << "param = " << typeid(param).name() << endl;
}

std::vector<Widget> createVec(); // factory function

const auto vw =  createVec(); // init factory

if(!vw.empty()){
  f(&vw[0]); //call f
}
//上例中vc++ 中输出T的类型为 class Widget const *， param 的类型为class Widget const *,
//实际上 T类型为class Widget const *, param的类型为class Widget const * &
//这是由于type_info::name授权类型使用传值方式，因此在处理时会将引用和const都去掉!!
//上面这个例子用IDE等显示也是不能正常处理!!
~~~

&ensp; 要正确处理<<Effective Modern c++>>中指出需要使用boost的```type_index.hpp```中的```boost::typeidex::type_id_with_cvr```
顾名思义with_cvr为不忽略const 、volatile和reference。

> 这里想起了一句话： 人生苦短，我用python，好尴尬 ( ¯(∞)¯ )

**总结**

- 通过上面例子可以看出c++在显示deduced type时对于简单的类型可以正常处理，但是比较负责时可能误导用户。感觉还是比较鸡肋 (⊙﹏⊙)

- 用类型推导时最好不要搞得太复杂，即使前面模板推导原理记得很清楚，还是容易掉坑。
