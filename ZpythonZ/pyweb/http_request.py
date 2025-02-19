# -- coding: utf-8 --
import json
import threading

from flask import Flask, request, make_response, jsonify

import time

app = Flask(__name__)

@app.route('/')
def home():
    return "python web test"

@app.route('/book', methods=['HEAD', 'GET', 'POST', "PUT", "PATCH", "DELETE", 'TRACE'])
def book():
    method = request.method
    if method == 'HEAD':
        return book_head()
    elif method == 'GET':
        return book_get()
    elif method == 'POST':
        return book_post_form()
        # return book_post_json()
    elif method == 'PUT':
        return book_put()
    elif method == 'DELETE':
        return book_delete()

def book_head():
    return jsonify({'name': 'python', 'star': 9.1})

def book_get():
    return jsonify({'name': 'python', 'star': 9.1})

def book_post_form():
    # name = request.form['name']
    # price = float(request.form['star'])
    name = request.form.get('name')
    price = float(request.form.get('score'))
    print(name, price)

    return jsonify({'status': 1, 'msg': 'form 新增成功'})

def book_post_json():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': 'json 新增成功'})

def book_put():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': '修改成功'})



def book_delete():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': '删除成功'})

def run_app():
    app.run(debug=False, port=5000)

'''
    # 
    ##notenote
    app.run(debug=True, port=5000)
    在 Flask 应用中，app.run() 是一个阻塞调用。
    这意味着当 app.run() 被执行时，它会启动 Flask 服务器，并且在服务器运行期间阻塞代码的进一步执行。
    换句话说，只要 Flask 应用仍在运行，位于 app.run() 调用之后的代码就不会被执行。
    因此，test_options() 函数在你的设置中根本就没有机会被调用。
    ## 但是也可以通过创建一个线程来启动Flask应用，而主线程用于执行测试
'''



if __name__ == '__main__':

    import requests

    t = threading.Thread(target=run_app)
    # #note1#
    # 因该将run_app函数作为目标传递给Thread对象
    # #error1#
    # t = threading.Thread(target=run_app())
    # # 在这里，run_app() 函数被调用并执行了，而不是将它作为函数引用传递给 Thread 类。
    # # 这意味着 app.run() 在主线程中执行，而不是在新线程中。
    # # 由于 app.run() 是阻塞调用，它会阻止代码继续执行到创建线程和发送请求的部分，
    # # 直到 Flask 服务器停止。
    t.start()
    # 创建并启动一个新的线程 t，目标函数是 run_app。
    # 这个 run_app 函数实质上是启动 Flask 应用的 app.run() 调用。

    time.sleep(2)
    # 这行代码使主线程暂停2秒钟。
    # 这个延时是为了给 Flask 应用足够的时间启动并准备接受连接。
    # 这是必要的，因为服务器可能需要一些时间来完全启动并开始监听端口。

    # # test OPTION
    # url = 'http://127.0.0.1:5000/book'
    # res = requests.options(url)   # 使用requests库向Flask 应用发送一个 HTTP OPTIONS 请求。
    # print(res.headers)
    # t.join()   # 等待 t 线程（即 Flask 应用的线程）结束。
    # 在实际应用中，这通常发生在服务器被关闭或出现错误时。
    # 在这个测试代码的情况下，如果不手动停止 Flask 服务器，t.join() 会一直阻塞。

    # #question#
    # url 页面404 not found
    # 表明你尝试访问的 URL 在 Flask 应用中没有对应的路由处理函数。
    # 请求是发送到根 URL / （即 GET / HTTP/1.1），
    # Flask 应用中只定义了 /book 路径的处理。
    # 因此，当请求 / 路径时，服务器找不到匹配的路由，就返回了 404 Not Found 错误。
    # #notenote#
    # 定义一个def home() 就好了。

    # # OPTIONS
    # url = 'http://127.0.0.1:5000/book'
    # def test_options():
    #     res = requests.options(url)
    #     print(res.headers)
    # test_options()
    # # {'Server': 'Werkzeug/3.0.1 Python/3.10.13', 'Date': 'Sat, 13 Apr 2024 10:27:27 GMT', 'Content-Type': 'text/html; charset=utf-8', 'Allow': 'OPTIONS, HEAD, POST, TRACE, GET, DELETE', 'Content-Length': '0', 'Connection': 'close'}

    # GET
    # url = 'http://127.0.0.1:5000/book'
    # def test_get():
    #     params = {'name': 'python'}
    #     res = requests.get(url, params=params)
    #     print(res.text)
    # test_get()
    # # {"name":"python web","star":9}

    # # HEAD
    # url = 'http://127.0.0.1:5000/book'
    # def test_head():
    #     params = {'name': 'python'}
    #     res = requests.head(url, params=params)
    #     print(res.text)   # "HEAD /book?name=python HTTP/1.1" 200 -  并不返回东西
    # test_head()

    # POST

    # url = 'http://127.0.0.1:5000/book'
    # def test_form_post():
    #     data = {'name': 'python', 'score': 45.6}
    #     # 这个方法好用  eval(x)  就是执行x
    #     data = eval(str(data))
    #     res = requests.post(url, data=data)
    #     print(res.text)
    #
    # test_form_post()
    # # error

    # url = 'http://127.0.0.1:5000/book'
    # def test_json_post():
    #     data = {'name': 'python', 'price': 45.6}
    #     res = requests.post(url, json=data)
    #     print(res.text)
    #
    #
    # test_json_post()

    # url = 'http://127.0.0.1:5000/book'
    # def test_put():
    #     data = {'name': 'python', 'price': 55.6}
    #     res = requests.put(url, json=data)
    #     print(res.text)
    # test_put()
    # ## --功能没写全
    # url = 'http://127.0.0.1:5000/book'
    # def test_patch():
    #     data = {'name': 'python'}
    #     res = requests.request('trace', url, json=data)
    #     print(res.text)
    # test_patch()

    url = 'http://127.0.0.1:5000/book'
    def test_delete():
        data = {'name': 'python', 'price': 55.6}
        res = requests.delete(url, json=data)
        print(res.text)
    test_delete()

    # # ## --功能没写全
    # url = 'http://127.0.0.1:5000/book'
    # def test_trace():
    #     data = {'name': 'python'}
    #     res = requests.request('trace', url, json=data)
    #     print(res.text)
    # test_trace()

    t.join()
