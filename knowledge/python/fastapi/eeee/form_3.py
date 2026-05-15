from urllib import parse
import requests

data = {
    'username': 'lili',
    'password': 'pd'
}

data = parse.urlencode(data)
# 因为是上传form表单，所以headers需要指定
headers = {"Content-Type":"application/x-www-form-urlencoded"}

res = requests.post("http://127.0.0.1:8000/login", data=data, headers=headers)
print(res.json())