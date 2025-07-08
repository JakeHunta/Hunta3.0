from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Allow all CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor).squeeze().cpu().numpy()
    return features.flatten()

class Listing(BaseModel):
    title: str
    image: str
    price: str
    link: str
    score: float

def fetch_ebay_results(term: str):
    EBAY_APP_ID = "JakeJaco-Hunta-PRD-bf018073b-70bec527"
    url = "https://svcs.ebay.com/services/search/FindingService/v1"
    params = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.0.0",
        "SECURITY-APPNAME": EBAY_APP_ID,
        "RESPONSE-DATA-FORMAT": "JSON",
        "REST-PAYLOAD": "true",
        "keywords": term,
        "paginationInput.entriesPerPage": 10
    }

    resp = requests.get(url, params=params).json()
    return resp.get("findItemsByKeywordsResponse", [{}])[0].get("searchResult", [{}])[0].get("item", [])

@app.get("/search", response_model=List[Listing])
def search(term: str = Query(...)):
    print(f"[SEARCH] Term: {term}")
    listings = fetch_ebay_results(term)
    text_features = extract_features(Image.new("RGB", (224, 224)))
    results = []

    for item in listings:
        title = item.get("title", [""])[0]
        image_url = item.get("galleryURL", [""])[0]
        price = item.get("sellingStatus", [{}])[0].get("currentPrice", [{}])[0].get("__value__", "N/A")
        view_url = item.get("viewItemURL", [""])[0]
        score = 0.0

        if image_url:
            try:
                image_data = requests.get(image_url, timeout=5)
                image = Image.open(BytesIO(image_data.content)).convert("RGB")
                img_features = extract_features(image)
                score = float(cosine_similarity([text_features], [img_features])[0][0])
            except Exception as e:
                print(f"[Image Error] {e}")

        results.append({
            "title": title,
            "image": image_url,
            "price": price,
            "link": view_url,
            "score": round(score, 3)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

@app.get("/")
def health():
    return {"status": "Hunta backend is alive"}
