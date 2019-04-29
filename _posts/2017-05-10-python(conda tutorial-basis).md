---
layout: post
title: conda tutorial-basis
date: 2017-05-10 16:00
description: 本篇文章主要介绍python包管理工具miniconda的使用
tag: python
---

## **引言**

&ensp; 由于近期需要把<machine learning action>这本书看掉，并参照书中python代码顺便学习一下python这门语言。
但是这本书的代码python版本为2，而python3近来越来越普及，而且优势也比较明显。因此综合考虑需要一个管理python环境和
python包入手快的工具，网上很多推荐用anaconda，然而安装过大，很多anaconda默认的包虽然很多人用，但是不适合初学者。最后
选择miniconda，该初始时仅仅包含一些conda依赖的包和pip包(保留pip主要是有些package，用conda无法安装)。下面简单
记录一下conda基本的操作命令，其实这个工具如果git用的熟，会发现很多思想异曲同工 👍

> 虽然本人近期转到windows平台下，但是未指明特殊平台的命令，全平台通用。

## **1. 安装 conda**

下载最新版[conda](https://conda.io/miniconda.html)，其中默认python为3.x，后面可以更新到指定版本

**_注意——**

下面开始介绍conda常用的命令，这里需要注意的是：

- 一些package如```pip```、 ``` python ```可以有自己独立的命令，
： ``` pip install numpy ``` 和 ``` python --version ```，其他一般的包是无法像 ``` numpy --version ```这样操作的!
一般的命令都是 ``` conda xxxx ```开头。

- 在两个破折号后面的options可以用缩写，eg： ``` --name ```等价于 ``` -n ```； ``` -envs ```等价于 ``` -e ```

## **2. 管理 conda**

| 功能 | 命令     |
| :------------- | :------------- |
| 验证conda版本       | ```conda --version```       |
| 更新conda到最新版本       |   ``` conda update conda ```     |


## **3. 管理环境**

&ensp; conda的环境有点像git的版本，当前处于某个环境将会在该环境名前面带\*，如下所示：

![图1-1](/images/posts/python/image1_1.PNG)

| 功能 | 命令     |
| :------------- | :------------- |
|  创建新环境      |   ``` conda create --name snowflakes python=2 numpy ``` 这里将创建环境/env/snowflakes ，并且该环境python版本为2，安装了numpy    |
|  激活环境（切换环境）      |   ``` activate snowflakes ```     |
|  退出该环境（返回root环境）     |   ``` deactivate ```，从A环境到B环境可以直接```activate B``` |
|  列出所有环境      |   ``` conda info --envs  ```     |
|  拷贝A环境到B     |   ``` conda create --name B --clone A ```     |
|  删除环境A     |   ``` conda remove --name A --all ```     |

## **4. 管理python**

| 功能 | 命令     |
| :------------- | :------------- |
| 查看所有可以安装的python版本       | ``` conda search --full-name python ```       |
| 查看当前环境python版本      |   ``` python --version ```     |

## **5. 管理packages**

| 功能 | 命令     |
| :------------- | :------------- |
|  显示当前环境安装的packages      | ``` conda list  ```       |
|  查找一个package     | ``` conda search numpy  ```，此时将显示所有可以安装的带numpy字段的包，若要完全匹配，则 ``` conda search --full-name numpy ```       |
|  安装一个新的package     | ``` conda install --name bunnies beautifulsoup4  ```，指定在环境bunnies下安装beautifulsoup4，若要在当前环境安装，则可以不用 ``` --name bunnies ```，可以指定packages版本： ``` conda install --name bunnies beautifulsoup4 ```     |
|  从anaconda下载packages     | ``` conda install --channel https://conda.anaconda.org/pandas bottleneck  ``` 从anaconda下载bottleneck，或者 ``` conda install --channel pandas bottleneck ```      |
|  用pip安装/卸载packages     | ``` pip install see  ```，若conda没有可用的包，则可以从pip用pip下载，由于conda中没有该packages，故卸载该packages也要用pip卸载 ``` pip uninstall see ```      |

|  卸载package    |   ``` conda remove --name bunnies numpy ```，从bunnies环境卸载numpy，若从当前环境卸载，可以不用 ``` --name bunnies ```     |

&ensp; 上面说了安装package可以用 ``` pip install xxx ```或者 ``` conda install xxx ```，当装packages后，若该package为pip管理，则会在package后面
提示：

![图1-2](/images/posts/python/image1_2.PNG)

如上，see为pip安装的，故在后面有 ``` <pip> ```符号。

## **6.导出与导入环境**

| 功能 | 命令     |
| :------------- | :------------- |
| 导出环境到environment.yaml文件       | 首先切换到需要导出的环境，然后 ``` conda env export > environment.yml ```       |
| 从文件创建一个环境     |   ``` conda env create -f environment.yml ```     |
| 建立一个独立的conda环境     |   有时我们为了控制每个package在一个环境中具体的版本号，需要创建一个独立的conda环境，这样当导出该环境到文件后，其他用户导入后会按照相同的packages版本来构建环境。因此使用这种方式一般是确定了要用到该环境的地方和本工作环境系统一致如都是win64：```conda list --explicit > spec-file.txt  ```     |
| 从文件读取一个独立环境     |   ``` conda create --name MyEnvironment --file spec-file.txt ``` 使用spec-file中的packages创建一个MyEnvironment环境或者 ``` conda install --name MyEnvironment --file spec-file.txt ```增加spec-file.txt中包含的特定packages  |

&ensp; 导出的文件默认是保存在 ``` C:\Users\user_name ```，user_name为你windows用户名。
environment.yml文件格式如下：

~~~
name: bunnies
channels:
- defaults
dependencies:
- beautifulsoup4=4.5.3=py27_0
- bottleneck=1.2.0=np112py27_0
- mkl=2017.0.1=0
- numpy=1.12.1=py27_0
- pip=9.0.1=py27_1
- python=2.7.13=0
- setuptools=27.2.0=py27_1
- vs2008_runtime=9.00.30729.5054=0
- wheel=0.29.0=py27_0
prefix: D:\Miniconda3\envs\bunnies
~~~

spec-file.txt文件的格式一般如下：

~~~
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: win-64
@EXPLICIT
https://repo.continuum.io/pkgs/free/win-64/beautifulsoup4-4.5.3-py27_0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/bottleneck-1.2.0-np112py27_0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/mkl-2017.0.1-0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/numpy-1.12.1-py27_0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/pip-9.0.1-py27_1.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/python-2.7.13-0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/setuptools-27.2.0-py27_1.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/vs2008_runtime-9.00.30729.5054-0.tar.bz2
https://repo.continuum.io/pkgs/free/win-64/wheel-0.29.0-py27_0.tar.bz2
~~~

## **保存环境变量**

假设你有一个环境analytics，并且存储可一个登入服务器的secret key和配置文件的路径，则我们可能需要
一个名为 ``` env_vars ```的脚本来分别执行赋值与清除。

**Linux and macOS**

~~~
cd /home/jsmith/anaconda3/envs/analytics
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
~~~

编辑 ```./etc/conda/activate.d/env_vars.sh```

~~~
#!/bin/sh
export MY_KEY='secret-key-value'
export MY_FILE=path/to/my/file
~~~

编辑 ```./etc/conda/deactivate.d/env_vars.sh```

~~~
#!/bin/sh

unset MY_KEY
unset MY_FILE
~~~

则当你使用 ``` source activate analytics ```环境变量MY_KEY和MY_FILE将会被赋值，同时当你 ``` source deactivate ```后这两个变量将会被清除。

**Windows**

~~~
cd C:\Users\jsmith\Anaconda3\envs\analytics
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
type NUL > .\etc\conda\activate.d\env_vars.bat
type NUL > .\etc\conda\deactivate.d\env_vars.bat
~~~

编辑 ```.\etc\conda\activate.d\env_vars.bat```

~~~
set MY_KEY='secret-key-value'
set MY_FILE=C:\path\to\my\file
~~~

编辑 ```.\etc\conda\deactivate.d\env_vars.bat```

~~~
#!/bin/sh
set MY_KEY=
set MY_FILE=
~~~

则当你使用 ``` activate analytics ```环境变量MY_KEY和MY_FILE将会被赋值，同时当你 ``` deactivate ```后这两个变量将会被清除。

## **8. 卸载conda**

- windows卸载直接从控制面板中卸载conda

- linux、os x : ```rm -rf ~/miniconda```

**参考：**

[1] conda 官方教程

[1]: https://conda.io/docs/test-drive.html#managing-conda "conda managing"
