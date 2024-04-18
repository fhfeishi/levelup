from fastapi import FastAPI, Form
import uvicorn
app = FastAPI()

@app.post('/login')
def login(username: str = Form(...), password: str = Form(...)):
    return {'username': username, 'password': password}


if __name__ == '__main__':

    uvicorn.run("eee:app", host="0.0.0.0", port=8000, log_level="info")
