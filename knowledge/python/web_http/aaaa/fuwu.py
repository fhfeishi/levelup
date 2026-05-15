import json
from flask import Flask, request, make_response, jsonify


app = Flask(__name__)


@app.route("/book", methods=['HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'TRACE'])
def book():
    method = request.method
    if method == 'HEAD':
        return book_head()
    elif method == 'GET':
        return book_get()
    elif method == 'POST':
        #return book_post_form()
        return book_post_json()
    elif method == 'PUT':
        return book_put()
    elif method == 'DELETE':
        return book_delete()

def book_head():
    return jsonify({'name': 'python进阶教程', 'price': 35.5})


def book_get():
    return jsonify({'name': 'python进阶教程', 'price': 35.5})


def book_post_form():
    name = request.form['name']
    price = float(request.form['price'])
    print(name, price)

    return jsonify({'status': 1, 'msg': '新增成功'})


def book_post_json():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': '新增成功'})


def book_put():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': '修改成功'})


def book_delete():
    data = json.loads(request.get_data())
    print(data)
    return jsonify({'status': 1, 'msg': '删除成功'})



if __name__ == '__main__':
    app.run(debug=True)