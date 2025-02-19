from fastapi import FastAPI

app = FastAPI()

@app.get("/")
# app.get 是一个装饰器，用于装饰index函数，处理path为/ 的请求
def index():
    return {'msg': "fastapi start!"}
#  index函数返回一个字典，FastAPI会自动将其转为json字符串

# 命令行启动   --适用于生产环境
# uvicorn main:app --reload
# 前面创建的脚本是main.py，创建的FastAPI的实例是app，使用--reload 当脚本发生变化时可以自动加载，想了解uvicorn 如何使用，可以通过
# uvicorn main:app --reload