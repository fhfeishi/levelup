import requests

data = {
    'name': '小明',
    'age': 20,
    'address': '北京',
    'family': {
        'father': '小明爸爸',
        'mother': '小明妈妈'
    },
    'school':{
        'name': '育英中学',
        'address': '育英路'
    },
    'course': [{'name': '语文', 'score': 100}, {'name': '数学', 'score': 100}]

}
res = requests.post("http://127.0.0.1:8000/users", json=data)
print(res.json())