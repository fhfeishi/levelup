import requests

url = 'http://127.0.0.1:5000/book'


def test_options():
    res = requests.options(url)
    print(res.headers)

test_options()

# {'Server': 'Werkzeug/3.0.1 Python/3.10.13', 'Date': 'Sat, 13 Apr 2024 12:46:58 GMT', 'Content-Type': 'text/html; charset=utf-8', 'Allow': 'GET, HEAD, DELETE, POST, OPTIONS, TRACE, PUT', 'Content-Length': '0', 'Connection': 'close'}


##########################################
## ---------------OPTIONS------------------------ ##
# * 用途：此方法用于描述目标资源的通信选项，通过发送一个预检请求来确定服务器支持的方法。
# * 场景：在CORS（跨源资源共享）中，浏览器可以先发送一个OPTIONS请求来检查实际请求是否安全可接受。
# * 向服务器发送options方法，可以测试服务器功能是否正常，服务器会返回这个资源所支持的HTTP请求方法，在javascript中，使用XMLHttpRequest对象进行CORS跨域资源共享时，会先使用options方法进行嗅探，以此判断对指定资源是否具有访问权限。
# * flask框架会自动处理OPTIONS和HEAD请求，我在指定'/book'所支持的methods中并没有写OPTIONS，但使用requests发送OPTIONS请求，可以得到正常响应
# * response 的headers里，会返回Allow 首部，其内容为"TRACE, GET, HEAD, PATCH, POST, DELETE, OPTIONS, PUT"，这表示，请求'/book'时，服务器支持这么多的请求方法。

