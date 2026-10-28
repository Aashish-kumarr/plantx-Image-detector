import sys
import os
import requests
import io
import json
from pathlib import Path
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

def make_green_leaf_image() -> bytes:
    img = Image.new("RGB", (640, 480), color=(34, 139, 34))
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=95)
    return b.getvalue()

def run_http_single_condition_test():
    green_bytes = make_green_leaf_image()
    
    # Upload
    up = requests.post(
        f"{BACKEND_URL}/images",
        files={"image": ("single_cond_test.jpg", io.BytesIO(green_bytes), "image/jpeg")},
        timeout=30
    )
    assert up.status_code == 201, f"Upload failed: {up.text}"
    image_id = up.json()["image_id"]
    
    # Analyze
    an = requests.post(
        f"{BACKEND_URL}/analyses",
        json={"image_id": image_id},
        timeout=120
    )
    assert an.status_code == 201, f"Analyze failed: {an.text}"
    result = an.json()
    analysis_id = result["analysis_id"]
    
    # Get from DB via HTTP
    get_res = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}", timeout=30)
    assert get_res.status_code == 200, f"GET analysis failed: {get_res.text}"
    db_doc = get_res.json()
    
    print("\n--- REAL HTTP SINGLE-CONDITION TEST ---")
    print(f"Analysis ID: {analysis_id}")
    print(f"HTTP Status: {an.status_code}")
    print(f"Pipeline Status: {db_doc['status']}")
    if db_doc.get("prediction"):
        pred = db_doc["prediction"]
        print(f"Crop: {pred.get('crop')}")
        print(f"Disease Supported: {pred.get('disease_supported')}")
        print(f"Class ID: {pred.get('class_id')}")
        print(f"Display Name: {pred.get('display_name')}")
        print(f"Score: {pred.get('score')}")
        print(f"Model Route: {pred.get('model_route')}")
        print(f"Disease Source: {pred.get('disease_source')}")
    print("HTTP SINGLE-CONDITION TEST PASS!")

if __name__ == "__main__":
    run_http_single_condition_test()
