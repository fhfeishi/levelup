from fastapi import FastAPI
import uvicorn
from typing import Optional, List
from pydantic import BaseModel

app = FastAPI()

# class User(BaseModel):
#     name: str       # 姓名
#     age: int        # 年龄
#     address: Optional[str] = None       # 住址 可选字段
#
#
# @app.post('/users')
# def add_user(user: User):
#     return {'msg': f'recv user info ,name is {user.name}'}

class SchoolInfo(BaseModel):
    name: str      # 学校名称
    address: str   # 学校地址


class FamilyInfo(BaseModel):
    father: str    # 父亲姓名
    mother: str    # 母亲姓名

class Course(BaseModel):
    name: str       # 课程名称
    score: float    # 分数

class User(BaseModel):
    name: str       # 姓名
    age: int        # 年龄
    address: Optional[str] = None       # 住址 可选字段
    family: FamilyInfo      # 家庭信息
    school: SchoolInfo      # 学校信息
    course: List[Course]    # 课程信息


@app.post('/users')
def add_user(user: User):
    return user.json()


if __name__ == '__main__':
    
    uvicorn.run("e:app", host="0.0.0.0", port=8000, log_level="info")
