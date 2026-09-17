import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request

"""
在请求得到路径操作处理之后，正式返回给客户端之前，
如果你想对response做一些操作，
仍然可以使用中间件，在Flask框架里通过after_request装饰器实现，

"""

app = FastAPI()

@app.middleware("http")
async def add_access_cookie(request: Request, call_next):
    print('add_access_cookie')
    response = await call_next(request)
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response.set_cookie(key='last_access_time', value=time_now)
    return response


@app.get('/index')
def index():
    return 'ok'


if __name__ == '__main__':
    uvicorn.run("jmidxxx:app", host='0.0.0.0', port=8000, log_level='info')

"""
每个请求在返回结果之前，都会在响应对象里设置cookie， 
实践中，有非常多的场景需要使用中间件，
比如在请求结束之后关闭数据库连接，
记录日志等等
"""
