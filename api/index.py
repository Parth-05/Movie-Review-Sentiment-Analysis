from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load your model from Hugging Face Hub
model_name = "Parth05/finetuning-sentiment-model-3000-samples"
sentiment = pipeline("text-classification", model=model_name)

app = FastAPI()

class Input(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment API is live!"}

@app.post("/predict")
def predict(inp: Input):
    res = sentiment(inp.text)
    return {"label": res[0]["label"], "score": res[0]["score"]}
