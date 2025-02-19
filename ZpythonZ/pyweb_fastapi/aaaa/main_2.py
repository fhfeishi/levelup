import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def index():
    """
    第一个API接口 \n
    :return:
    """
    return {"msg": "Hello World"}


if __name__ == '__main__':
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

    # 这些数字，如 54236、54392、62404 等，表示的是客户端（访问你的服务器的设备）使用的端口号。
    # 在网络通信中，端口号是一个数字标识，用于在一台设备上区分不同的网络服务或不同的网络连接。
    # 每一个网络连接都由一个IP地址和一个端口号组成的组合唯一标识。

    # 两种交互式文档,nb  ---自动配置好的,
    # http://ip:prot/docs
    # http://ip:prot/redoc

    # 使用交互式API文档，可以降低前后端的沟通成本和不同系统之间的对接成本，
    # 相比于规范的写在wiki或者飞书上的API文档，
    # 交互是API文档的一大优势就是它可以做到及时更新，现在，我为index函数添加函数文档

    # 为什么没有这段介绍  " 第一个API接口 "