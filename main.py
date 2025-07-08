from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Listing(BaseModel):
    title: str
    image: str
    price: str
    link: str
    score: float

@app.get("/search", response_model=List[Listing])
def search(term: str = Query(...)):
    return [
        {
            "title": "Used Strymon OB-1",
            "image": "https://example.com/image.jpg",
            "price": "£120",
            "link": "https://example.com/item",
            "score": 0.93
        }
    ]

@app.get("/")
def root():
    return {"status": "Hunta backend is alive."}
