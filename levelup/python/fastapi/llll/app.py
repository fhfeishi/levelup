import uvicorn
from fastapi import FastAPI
from starlette.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()
template = Jinja2Templates(directory='templates')       # 创建模板对象


@app.get('/welcome/{name}')
def welcome(name: str, request: Request):
    return template.TemplateResponse('welcome.html', {'name': name, 'request': request})

# 想要使用jinja2模板，在返回TemplateResponse对象时，必须传request， 否则就会报错。





if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, log_level='info')
