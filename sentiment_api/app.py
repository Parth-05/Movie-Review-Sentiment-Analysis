from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load your fine-tuned model from Hugging Face
model_name = "Parth05/finetuning-sentiment-model-3000-samples"
classifier = pipeline("text-classification", model=model_name)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment API is live!"}

@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}