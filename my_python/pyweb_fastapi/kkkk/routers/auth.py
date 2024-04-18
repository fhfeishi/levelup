from fastapi import APIRouter

auth_router = APIRouter()

@auth_router.get('/login')
def login():
    return '登录功能'

@auth_router.get('/logout')
def logout():
    return '退出登录'

@auth_router.get('register')
def register():
    return '注册功能'