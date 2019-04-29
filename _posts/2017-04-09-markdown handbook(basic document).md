---
layout: post
title: (1)markdown基本语法速记
date: 2017-04-09 20:10
description: 这篇博客主要是对markdown常用语法做个总结
tag: markdown
---

## **前言**
<p style="color:red;font-size:3.em;front-weight:bold">由于GitPages支持kramdown，
且这个解释器支持MathJax(基本语法和Latex一致)，故本篇只对kramdown支持的
语法进行记录!
</p>

#  **markdown 基本语法**

## **标题**
+ 在文字前面加上1-6个```#```即可生存html对应的h1-h6标题

## **改变字体样式**
+ 在文字左右两边加上&ensp; ```**``` &ensp; 即可实现加粗字体如： &ensp; ``` **这是粗体** ``` &ensp;->&ensp; **这是粗体**
+ 文字两边加上&ensp; ```_```&ensp;可将字体变为斜体如： &ensp; ```_这是斜体_```&ensp;->&ensp;_这是斜体_
+ 以上两个可以嵌套使用，变为斜粗体如:&ensp; ```**_这是斜粗体_**```&ensp;->&ensp;**_这是斜粗体_**

## **列表**

+ 使用```* ```，```+ ```，```- ```分别+空格，将编程如本行就是一个列表，列表头是一个点
1. 有序表可以用```数字+英文点+空格```，如本列表
    1. 上面两个支持嵌套列表效果(每一级分别相对前一级增加四个空格)

## **引用**
+ 在欲引用的文字前面加上```>+空格```
+ 若要两级嵌套需要换行并在开头```>+空格>+空格``` 即两个```>```空格，依次类推
+ 可以嵌套代码块 \>\> \`\`\` 代码 \`\`\` 如：当前(krmdown中无效)


## **代码块**
+ 不换行用 \`\`\`代码\`\`\`

```
def what?
  42
end
```


+ 保留原来格式用```~~~代码~~~```

~~~ Ruby  
def what?
  42
end
~~~

+ 使用 ```<pre>代码</pre> ```元素同样可以形成代码块  
<pre>
def what?
  42
end
</pre>

## **链接**
+ ```[描述](url "可选标题")```，如:&ensp; [博客主页](http://zbabby.com)
+ 不需描述，直接```<url>```，如:&ensp; <http://zbabby.com>
+ 先声明```[ref_name][url]```，再在文档的结尾添加url网址如```[url]: http://zbabby.com ``` 注意冒号:后有个空格 &ensp;如:      [zbabby][url1]    



## **图片**
+ ```![描述](url)```，其中url若是引用站内图片，最好用相对地址如：&ensp;```![cat](/img/posts/markdown doc/cat.jpg) ```

![cat](/img/posts/markdown doc/cat.jpg)

+ （当前）仅支持一种方式设置图片显示大小
即\!\[描述\]\(url\){:with="50px" height="50px"}
![cat](/img/posts/markdown doc/cat.jpg){:width="50px" height="50px"}

## **换行和空格**
+ 空格可以在欲插入空格的地方插入```&ensp;```注意分号;不能忘
+ 换行可以在上一段结尾后加两个空格；空出一行；在上一段结尾或者本段开头加```<br>```

## **分割线**
+ 在前后段之间加入三个减号```---```,且保证``---```和前后段之间至少空**两行**如:   

前段

---

后段  

  [url1]: http://zbabby.com
