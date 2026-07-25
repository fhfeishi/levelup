

"""
在使用FastAPI 开发项目时，不可能将所有路径操作都写在一个文件中，
那样不利于划分模块和维护代码，
FastAPI 提供了APIRouter 蓝图技术，
可以让你将功能紧密相连的代码写一个模块里，
合理的划分项目的结构。
"""

from fastapi import FastAPI
from routers.user import user_router
from routers.auth import auth_router
import uvicorn

app = FastAPI()
app.include_router(user_router)
app.include_router(auth_router)

if __name__ == '__main__':
    uvicorn.run("bluemap:app", host='0.0.0.0', port=8000, log_level='info')






