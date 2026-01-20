from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class Modelname(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

fake_name_db = [{"KJH":"13"},{"ZHH":"14"},{"KJHH":"15"}]

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 1):
    return fake_name_db[skip:skip+limit]

@app.get('/')
async def root():
    return {"message": "hello world"}

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "zhanghui is shit"}
        )
    return item

@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user_id(user_id: str):
    return {"user_id": user_id}

@app.get("/models/{model_name}")
async def get_model_name(model_name: Modelname):
    if model_name is Modelname.alexnet:
        return {"model_name": "Alexnet", "massage": "Deep learning FTW"}
    if model_name.value == "lenet":
        return {"model_name": "lenet"}
    return {"model_name": "resnet"}