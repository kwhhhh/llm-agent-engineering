from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    user: str | None = None

@app.get("/")
async def feed_back():
    return {"Hello": "World"}

