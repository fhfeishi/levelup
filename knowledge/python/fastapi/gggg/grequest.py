from fastapi import FastAPI, Request, Response
import uvicorn

app = FastAPI()


@app.get("/")
def index(request: Request):
    return {
        'path': request.url.path,           # 路径
        'port': request.url.port,           # 端口
        'method': request.method,           # 请求方法，
        'remote_ip': request.client.host,   # 客户端ip
        'headers': request.headers          # headers
    }

@app.post("/post")
async def post(request: Request):
    body = await request.body()
    print(body, type(body))
    post_data = await request.json()
    return post_data

if __name__ == '__main__':
    uvicorn.run("grequest:app", host='0.0.0.0', port=8000, log_level="info")


    """
    Request是FastAPI的请求对象，它代表一次具体的请求，包含了本次请求的一切信息，FastAPI通过在路径操作函数里进行类型标注可以获得请求的具体信息，例如query部分的参数，请求体，cooke，headers， 但某些情况下，直接从request对象里获取数据仍是一个快捷的方法。

    不同于Flask 的写法，FastAPI 的请求对象必须在路径操作定时作为函数的参数进行声明
    
    在定义路径操作函数时，标注request参数的类型是Request， 在函数里，使用request对象可以获得有关请求的一切信息。对于post等有body体的请求，也可以通过request对象来获取
    
    想要直接获得到请求的body，需要使用request.body()方法，它是一个协程，因此路径操作函数也必须定义为协程
    
    得到的body 是bytes类型数据，是没有经过任何加工的二进制数据，这一点可以通过print语句输出的结果证明。FastAPI 通过body方法直接获得请求的body在联调程序时非常有用处，系统在对接时，经常出现一方坚持认为发送的数据是正确的，另一方认为自己处理数据的逻辑是正确的，这个时候，通过body方法获得到最原始的未经过框架加工过的数据就显的尤为重要了，配合上headers里的首部信息，就容易定位问题出在哪里
    
    """