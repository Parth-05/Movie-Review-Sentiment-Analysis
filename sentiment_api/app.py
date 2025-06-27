import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Load model
model_name = "Parth05/finetuning-sentiment-model-3000-samples"
classifier = pipeline("text-classification", model=model_name)

app = FastAPI()

#  Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "")
origins = cors_origins.split(",") if cors_origins else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment API is live!"}

@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}
