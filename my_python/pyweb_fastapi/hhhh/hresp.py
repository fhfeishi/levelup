import uvicorn
from fastapi import FastAPI, Response

app = FastAPI()

# @app.get('/index')
# def index(response: Response):
#     response.set_cookie(key='user_type', value='old')
#     response.headers['token'] = 'server-token'
#     return 'ok'

@app.get('/index')
def index():
    response = Response(content='ok', media_type='text/html')
    return response

if __name__ == '__main__':
    uvicorn.run("hresp:app", host='0.0.0.0', port=8000, log_level="info")
