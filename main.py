import uvicorn  # ASGI
from fastapi import FastAPI

# Create the app object
app = FastAPI()

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome'}

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome Aboard, Captain': f'{name}'}

# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn main:app --reload
