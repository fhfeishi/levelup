import requests
res = requests.post("http://127.0.0.1:8000/post")
print(res.text)