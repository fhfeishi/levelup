from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'python web test zz'

@app.route('/book', methods=['GET', 'POST', 'DELETE'])
def book():
    method = request.method
    if method == 'GET':
        return jsonify({'name': 'python web', 'star': 9})
    elif method == 'POST':
        return jsonify({'status': 1, 'msg': 'Book added successfully'})
    elif method == 'DELETE':
        return jsonify({'status': 1, 'msg': 'Book deleted successfully'})

def test_flask_app():
    with app.test_client() as client:
        # 测试 GET 请求到根路径 '/'
        response = client.get('/')
        print('Response to GET /:', response.data.decode())

        # 测试 GET 请求到 '/book'
        response = client.get('/book')
        print('Response to GET /book:', response.json)

        # 测试 POST 请求到 '/book'
        response = client.post('/book')
        print('Response to POST /book:', response.json)

        # 测试 DELETE 请求到 '/book'
        response = client.delete('/book')
        print('Response to DELETE /book:', response.json)

if __name__ == '__main__':
    test_flask_app()

# ## test_flask_app 函数：这个函数使用 app.test_client() 创建了一个测试客户端，
# 它可以模拟向 Flask 应用发送 HTTP 请求。
# ## 测试不同的请求：函数中的代码模拟发送 GET、POST 和 DELETE 请求到 /book 路由，
# 以及一个 GET 请求到根路由 /。
# ## 打印响应：每个请求的响应数据被获取并打印出来，以便你可以看到每种请求类型的结果。

#### 这种方式可以在不启动 Flask 服务器的情况下测试你的应用逻辑，非常适合自动化测试。
