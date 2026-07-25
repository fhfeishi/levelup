import requests

data = {
    'name': 'lili',
    'age': 20,
    'address': '北京'
}
res = requests.post("http://127.0.0.1:8000/users", json=data)
print(res.text)

#  {"msg":"recv user info ,name is lili"}