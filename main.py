import os

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

import requests
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse
from pydantic import BaseModel, AnyUrl
from typing import Optional
from dotenv import load_dotenv

def fetch_body300(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)
        return text[:300]
    except Exception as e:
        return ""
        
device = torch.device("cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
model = DistilBertForSequenceClassification.from_pretrained("./model")
model.to(device)
model.eval()

def classify_url(url):
    body = fetch_body300(url)

    text = f"""
    URL: {url}
    Body: {body}
    """

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        )
        pred_id = outputs.logits.argmax(dim=1).item()

    return model.config.id2label[pred_id]


load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
ROOT_PATH = os.getenv("ROOT_PATH", "").strip()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not set. Set the API_KEY environment variable.",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return api_key


class PredictRequest(BaseModel):
    url: AnyUrl


class PredictResponse(BaseModel):
    label: str


app = FastAPI(title="URL Classifier API", root_path=ROOT_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def demo_page():
    return FileResponse("demo.html")


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    label = classify_url(str(req.url))
    return PredictResponse(label=label)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
