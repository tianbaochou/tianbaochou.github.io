---
layout: post
title: Github Pages 构建博客 (1)
date: 2019-04-29 20:10
description: 这篇博客主要是对github pages构建进行简单的介绍
tag: markdown
---

## **前言**
<p style="color:red ;font-size:3.em;front-weight:bold">模板来源： https://github.com/Huxpro/huxpro.github.io
</p>

## 1. 准备工具

+ jekyll [1](https://jekyllrb.com/docs/)

+ `gem update github-pages` 更新依赖包


## 2. 下载模板

+ ```bash
    git clone https://github.com/Huxpro/huxpro.github.io``` 

+  根据模板说明，在文件配置文件`_config.yml`替换相应的配置


## 4. 添加Gittalk

+  申请Github Application:

    + [注册一个Github Application](https://github.com/settings/applications/new)
    其中的`Application name`任意填写, `Homepage URL`和 `Authorization callback URL`需要填写你的博客的网址. <p style="color:red; font-size:3.em;font-weight:bold">注意当你绑定自定义域名后，需要填写自定义的域名，否则将无法正确初始化</p> 

    + 生成一个GitHub Application之后，网页会显示一个clientID和一个clientSecret,将其复制到代码中。其中这两个参数**只会显示一次**, 所以一定要记得填写完之后保存这两个参数

    + 添加`comments.html`到`_include/`文件夹下，添加如下内容: 
        ```html
        <section class="post-comments">
        <div id="gitalk-container"></div> <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/gitalk@1/dist/gitalk.css">
        <script src="https://cdn.jsdelivr.net/npm/gitalk@1/dist/gitalk.min.js"></script>
        <script>
            var gitalk = new Gitalk({
                    id: '{{ page.title }}',      // document.location.href // Ensure uniqueness and length less than 50{{ page.title }}
                    clientID: '{{ site.gitalk_clientID }}',
                    clientSecret: '{{ site.gitalk_Secret }}',
                    repo: '{{ site.gitalk_repo }}',
                    owner: '{{ site.gitalk_owner }}',
                    admin: ['{{ site.gitalk_admin }}'],
                    distractionFreeMode: '{{ site.distractionFreeMode }}'  // Facebook-like distraction free mode
                })
                gitalk.render('gitalk-container')
        </script>
        </section>

        ```
    + 在`_config.yml`中添加`gitalk_clientID`, `gitalk_Secret`, `gitalk_repo`, `gitalk_owner`, `gitalk_admin`, `distractionFreeMode`，这些参数说明如下：

    ```yaml
    gitalk_clientID: xxxxxxxx #申请的clientID
    gitalk_Secret: xxxxx      #申请的clientSecret
    gitalk_repo:              #用于评论的repo名，可以是托管博客的仓库名，也可以是另外建立的仓库名, eg xxx.github.io(必须为public),本文为custom_discuss(必须为public)
    gitalk_owner: xxxx        #用户名 
    gitalk_admin: xxxx        #用户名
    distractionFreeMode: True
    createIssueManually: True
    ```

    + 将`comments.html`插入到`_layout/post.html`中的`post-contianer`容器最下面
     
     ```
         <!-- Post Container -->
            <div class="
                col-lg-8 col-lg-offset-2
                col-md-10 col-md-offset-1
                post-container">

                ....
                     <!--gitalk-->
                {% include comments.html %}
            </div>
     ```

## 5. 添加latex支持

在`_include/head.html`的`<head> </head>`之间添加`mathjax`支持

```
    <!-- latex 支持 -->
    <script type="text/x-mathjax-config"> 
        MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); 
    </script>
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            processEscapes: true
            }
        });
    </script>
    
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript">
    </script>

```

## 6. 创建xxx.github.io
在github中创建github repo,其中项目名为`user_name.github.io`，`user_name`为你github的用户名，如果申请了域名，则绑定域名：

![域名](/img/posts/blogs/custom_domain.jpg)


## 9. 将clone下来的模板文件中的`.git`删除，并重新用git初始化，建立本地仓库和远程仓库的连接

```shell
rm -rf .git
git init
git commit -m"first commit"
git remote add origin https://github.com/use_name/user_name.github.io.git #注意替换user_name为你自己的用户名!!!!

```

## 7. 书写格式

在`_post/`文件夹下为我们每次编写的博客，支持markdown格式. 每次新建的文件名按`日期+标题`来规范，
比如`2018-04-09-Hello.md`. Markdown文件头需要有一段`yaml格式说明`,如下所示:


`2018-04-09-Hello.md`
```yaml
---
layout:     post
title:      "Hello"
subtitle:   "Hello World, Hello Blog"
date:       2015-01-29 12:00:00
author:     "xxx"
header-img: "img/post-bg-2015.jpg" # post背景图片
tags:
    - Life
---


### 这里是内容！！！！！！！

```