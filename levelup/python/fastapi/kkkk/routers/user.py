from fastapi import APIRouter

user_router = APIRouter(prefix='/users')

@user_router.get("/")
def all_users():
    return 'all users'

@user_router.get("/info/{uid}")
def user_info(uid: int):
    return {'uid': uid}