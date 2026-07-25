from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

"""
如果想在请求被特定的路径操作函数处理之前做一些操作，
例如检查发出请求的客户端IP是否符合白名单，
那么就需要在调用call_next 之前做一些事情，
这样的功能在Flask框架里是通过before_request 装饰器实现的，

"""
app = FastAPI()

@app.middleware("http")
async def check_remote_ip(request: Request, call_next):
    print('check_remote_ip')
    client_ip = request.client.host
    if client_ip == '127.0.0.1':
        return JSONResponse({"msg": 'bad request'})

    response = await call_next(request)
    return response


@app.get('/index')
def index():
    return 'ok'

if __name__ == '__main__':
    uvicorn.run("jmidxx:app", host='0.0.0.0', port=8000, log_level='info')

# check_remote_ip

"""
我定义一个检查客户端IP的中间件，如果ip不符合要求，立即返回检查结果，
不会使用index函数处理请求。需要注意的是，
在中间件里响应请求与在路径操作函数里响应请求是有不同的，
在路径操作函数里返回字典，FastAPI会自动帮你转成json数据，而在中间件里，
你必须自己实现这一点。

"""