# 动态路由
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# @app.get('/userinfo/{uid}')
# def user_info(uid):
#     return {'msg': f"recv uid :{uid} type is {str(type(uid))}"}

# # 为参数uid 增加类型标注，将其标注为int类型，FastAPI将根据这个标注对uid进行类型转换
# @app.get('/userinfo/{uid}')
# def user_info(uid: int):
#     return {'msg': f"recv uid :{uid} type is {str(type(uid))}"}

@app.get('/userinfo/all')
def user_info_all():
    return {'msg': "all user info"}

@app.get('/userinfo/{uid}')
def user_info(uid: int):
    return {'msg': f"recv uid :{uid} type is {str(type(uid))}"}

"""
路径 /userinfo/all 和 '/userinfo/{uid}' 是相似的，
如果请求/userinfo/all 被user_info函数 处理了，all不能转为int类型数据，
服务会报错。但实践中，这样写不会发生错误，，当请求/userinfo/all 到来时，
user_info_all 和 user_info 都可以用来处理请求，但user_info_all定义在前，
因此优先使用user_info_all 对请求进行处理。

"""

if __name__ == '__main__':
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

# http://127.0.0.1:8000/userinfo/5 