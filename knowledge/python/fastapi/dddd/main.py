from fastapi import FastAPI
from enum import Enum
import uvicorn

app = FastAPI()

class BookType(str, Enum):
    Novel = 'novel'             # 小说
    Economics = 'Economics'     # 经济类


@app.get('/books/{book_type}')
def book_info(book_type: BookType, page: int = 0, limit: int = 10):
    return {'msg': f'返回{book_type} 类型图书， 第{page}页， 每页{limit}条数据'}

# 函数book_info 里book_type 参数是和路径参数里的book_type相对应的，
# 对它的类型标注是BookType，是一个枚举类，如果传入的图书类型不在枚举范围内，
# FastAPI会自动报错。

if __name__ == '__main__':
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
