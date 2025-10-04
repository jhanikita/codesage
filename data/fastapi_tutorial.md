# FastAPI Quickstart

- FastAPI is a Python web framework for building APIs quickly.
- Install: `pip install fastapi uvicorn`
- Sample API:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello World"}

