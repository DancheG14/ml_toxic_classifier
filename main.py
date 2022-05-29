from fastapi import FastAPI
from transformers import pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english") as pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
