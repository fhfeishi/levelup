# 案例 01：个人资料卡

## 本课目标

制作一张属于自己的个人资料卡，并在浏览器开发者工具中观察它是怎样通过 HTTP 加载的。

这一课只关注三件事：

- HTML 描述页面的内容和结构。
- CSS 控制页面的视觉表现。
- 浏览器通过 HTTP 请求取得 HTML、CSS 等资源，再把它们渲染成页面。

暂时不使用 JavaScript。

## 开始前阅读

不需要从头读完整套教程，只阅读本案例用得到的部分：

- [HTML 基础](https://www.runoob.com/html/html-basic.html)
- [HTML 元素](https://www.runoob.com/html/html-elements.html)
- [HTML 属性](https://www.runoob.com/html/html-attributes.html)
- [CSS 语法](https://www.runoob.com/css/css-syntax.html)
- [CSS Id 和 Class 选择器](https://www.runoob.com/css/css-id-class.html)
- [CSS 盒子模型](https://www.runoob.com/css/css-boxmodel.html)
- [HTTP 教程](https://www.runoob.com/http/http-tutorial.html)只需先理解客户端、服务器、请求和响应。

## 第一步：运行起始页面

在当前目录打开终端：

```powershell
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```

不要直接双击 `index.html`。本课刻意使用本地 HTTP 服务，以便观察真实的请求和响应。

打开浏览器开发者工具：

1. 进入 `Network` 面板。
2. 刷新页面。
3. 找到 `index.html` 和 `style.css`。
4. 观察它们的请求方法、状态码和内容类型。

你应该能看到浏览器至少发出了两个 `GET` 请求，并收到成功响应。

## 第二步：完成 HTML 内容

在 `index.html` 的 `<main>` 中组织以下信息：

- 你的名字或昵称；
- 一句话介绍；
- 一个“正在学习”列表，至少包含三项；
- 两个链接，例如 GitHub、个人主页或常用技术网站。

要求：

- 页面只能有一个 `<h1>`；
- 学习列表使用 `<ul>` 和 `<li>`；
- 链接使用 `<a>`，并提供有效的 `href`；
- 根据内容含义选择元素，不要把所有内容都写成 `<div>`。

## 第三步：自由设计 CSS

修改 `style.css`，让资料卡在页面中清晰、易读。风格由你决定，但至少练习：

- class 选择器；
- `color` 和 `background-color`；
- `padding`、`margin` 和 `border`；
- `width` 或 `max-width`；
- 字体大小和行高；
- 一种用于排列或居中的布局方式。

视觉效果没有标准答案。可以简洁、复古、科技感或卡片风，但不要照抄教程截图。

## 功能验收

- 通过 `http://localhost:8000` 可以访问页面；
- 浏览器标签页显示了你设置的标题；
- 页面包含介绍、学习列表和链接；
- `style.css` 通过外部样式表方式加载；
- Network 面板能找到 HTML 和 CSS 请求；
- 缩窄浏览器窗口后没有明显横向滚动或内容遮挡；
- 暂时没有引入 JavaScript、框架或第三方组件库。

## 完成后思考

1. HTML 和 CSS 分别解决什么问题？
2. `class` 和 `id` 在使用场景上有什么区别？
3. `padding` 与 `margin` 分别位于盒子模型的什么位置？
4. 为什么页面打开时，浏览器还要单独请求一次 `style.css`？
5. 直接双击 HTML 与通过 `http://localhost:8000` 访问，有什么不同？
