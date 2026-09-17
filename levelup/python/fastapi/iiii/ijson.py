
# fastapi 返回json格式数据

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

class UserInfo(BaseModel):
    name: str
    age: int
    address: str


# @app.get('/user/{uid}', response_model=UserInfo)
# def user_info(uid: int):
#     return {'name': 'lili', 'age': 20, 'address': '北京', 'uid': uid}

# @app.get('/user/{uid}', response_model=UserInfo)
# def user_info(uid: int):
#     data = {'name': 'lili', 'age': 20, 'address': '北京', 'uid': uid}
#     user = UserInfo(**data)
#     return user

@app.get('/user/{uid}')
def user_info(uid: int):
    return JSONResponse(content={'name': 'lili', 'age': 20, 'address': '北京', 'uid': uid})

if __name__ == '__main__':

    uvicorn.run("ijson:app", host='0.0.0.0', port=8000, log_level="info")

