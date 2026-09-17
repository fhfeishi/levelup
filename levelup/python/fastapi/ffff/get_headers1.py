from typing import Optional
from fastapi import FastAPI, Form, Header, Request
import uvicorn

app = FastAPI()

# @app.get('/index')
# def index(user_agent: Optional[str] = Header(None)):
#     return {"User-Agent": user_agent}

@app.get('/index')
def index(request: Request):
    return {"headers": request.headers}


if __name__ == '__main__':

    uvicorn.run("get_headers1:app", host="0.0.0.0", port=8000, log_level="info")

    """
    许多header 使用-连接，比如User-Agent, 
    这种变量命名方式是不允许的，因此在定义参数时，
    需要以下划线来替代-, 此外，在处理header时支持大小写不敏感，
    想要获取User-Agent，在参数定义时写成user_agent就可以了，
    不能忽视的一点，一定要在设置默认值时显示标识Header。
    """