from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis",   
                      "distilbert-base-uncased-finetuned-sst-2-english")
@app.get("/")
async def root():
    return {"message": "Hello World"}


