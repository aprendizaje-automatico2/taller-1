from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def root():
    return {"message": "Hello World"}