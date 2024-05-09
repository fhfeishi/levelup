from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/get")
def get():
    """
    get method test
    :return:
    """  
    return {"msg": "recv get request"}

@app.post("/post")
def post():
    """
    post method test
    :return:
    """
    return {"msg": "recv post request"}

"""
在这个例子中，我定义了get和post两个函数，app.get 和 app.post 都是装饰器，
被这两个装饰器分别装饰后，get函数只能用来处理path 为 /get 且请求方法是GET的请求，同理，post函数
只能用来处理path 为 /post 且请求方法是POST的请求。
在Flask框架中，get和psot函数被称之为视图函数，而在FastAPI中，
被称之为操作路径函数，其实他们在本质上是一样的。

"""


"""
如果你希望一个函数即可以处理GET请求，也可以处理PSOT请求，则可以这样编写代码
@app.get("/test")
@app.post("/test")
def test_func():
    return {"msg": "recv http request"}

"""

if __name__ == '__main__':
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")




