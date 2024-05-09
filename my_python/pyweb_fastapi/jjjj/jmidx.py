# fastapi中间件
# FastAPI的中间件本质上是一个函数，
# 它在每个请求被特定的路径操作函数处理之前和之后工作，通过中间件，你可以在请求被处理之前或者之后做一些事情。

from fastapi import FastAPI, Request
import uvicorn
app = FastAPI()

@app.middleware("http")
async def before_request_1(request: Request, call_next):
    print('before_request_1')
    response = await call_next(request)
    return response


@app.middleware("http")
async def before_request_2(request: Request, call_next):
    print('before_request_2')
    response = await call_next(request)
    return response


@app.get('/index')
def index():
    return 'ok'

if __name__ == '__main__':
    uvicorn.run("jmidx:app", host='0.0.0.0', port=8000, log_level='info')

# before_request_2
# before_request_1
# 中间件的执行顺序与定义顺序是相反的。









