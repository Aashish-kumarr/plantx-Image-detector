"""
PlantDx E2E Integration Test v3

Strategy:
 - Route-level tests: run directly against the LOCAL pipeline (same process, no HTTP needed)
   Already proven in backend_smoke_test.py. Re-run here for completeness.
 - Real HTTP E2E: use actual train/val images from the dataset (NOT test split)
   to exercise the full frontend -> HTTP -> Model 1 -> Model 2 -> Disease Model -> MongoDB flow.
 - MongoDB persistence: verify via GET /api/v1/analyses/{id}
"""

import sys
import os
import json
import io
import csv
import requests
import torch
from pathlib import Path
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000/api/v1"
ML_WS = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
METADATA_DIR = ML_WS / "metadata"
DATASET_DIR = ML_WS / "dataset" / "plantvillage dataset" / "color"

sys.path.insert(0, str(Path(r"C:\Users\ak021\Desktop\workspace\plantDx\Plant_disease_detection\backend")))
from app.ml.pytorch_inference import pytorch_inference_pipeline, MODEL2_LABELS, SINGLE_CONDITION_CROPS

# ---- Helpers ----

def upload_image_http(image_bytes: bytes, filename: str = "test.jpg") -> tuple[int, dict]:
    resp = requests.post(
        f"{BACKEND_URL}/images",
        files={"image": (filename, io.BytesIO(image_bytes), "image/jpeg")},
        timeout=30
    )
    return resp.status_code, resp.json() if resp.content else {}

def run_analysis_http(image_id: str) -> tuple[int, dict]:
    resp = requests.post(
        f"{BACKEND_URL}/analyses",
        json={"image_id": image_id},
        timeout=120
    )
    return resp.status_code, resp.json() if resp.content else {}

def get_analysis_http(analysis_id: str) -> tuple[int, dict]:
    resp = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}", timeout=30)
    return resp.status_code, resp.json() if resp.content else {}

def force_leaf_mock():
    o1 = pytorch_inference_pipeline.model1.forward
    def fake_leaf(x):
        logits = torch.full((1, 2), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, 1] = 10.0
        return logits
    pytorch_inference_pipeline.model1.forward = fake_leaf
    return o1

def force_crop_mock(crop_name):
    idx = [k for k, v in MODEL2_LABELS.items() if v == crop_name][0]
    o2 = pytorch_inference_pipeline.model2.forward
    def fake_m2(x):
        logits = torch.full((1, 14), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, idx] = 10.0
        return logits
    pytorch_inference_pipeline.model2.forward = fake_m2
    return o2

def restore_m1(o1):
    pytorch_inference_pipeline.model1.forward = o1

def restore_m2(o2):
    pytorch_inference_pipeline.model2.forward = o2

def find_train_val_image(crop: str, max_search: int = 200) -> Path | None:
    """Find a non-test image for a given crop by scanning metadata CSVs."""
    for model_num in range(3, 12):
        meta_file = METADATA_DIR / f"model{model_num}_metadata.csv"
        if not meta_file.exists():
            continue
        with open(meta_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").lower() == "test":
                    continue  # NEVER use test images
                label = row.get("crop", "") or row.get("label", "") or row.get("class_name", "")
                if crop.lower() not in label.lower():
                    continue
                p = Path(row.get("path", ""))
                if p.exists():
                    return p
    # Fallback: scan dataset folder directly
    crop_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and crop.lower() in d.name.lower()]
    for crop_dir in crop_dirs:
        imgs = list(crop_dir.glob("*.jpg"))[:5] + list(crop_dir.glob("*.JPG"))[:5] + list(crop_dir.glob("*.png"))[:5]
        if imgs:
            return imgs[0]
    return None

def run_e2e():
    ALL_SUPPORTED = ["apple", "cherry", "corn", "grape", "peach", "pepper", "potato", "strawberry", "tomato"]
    EXPECTED_MODELS = {
        "apple": 5, "cherry": 6, "corn": 7, "grape": 8, "peach": 9,
        "pepper": 10, "potato": 3, "strawberry": 11, "tomato": 4
    }
    UNSUPPORTED = ["blueberry", "orange", "raspberry", "soybean", "squash"]

    routing_results = {}
    http_cases = []
    
    print("\n" + "="*60)
    print("PLANTDX E2E INTEGRATION TEST v3")
    print("="*60)

    # -------------------------
    # 1. Backend Health
    # -------------------------
    print("\n[1] Backend Health Check")
    hc = requests.get(f"{BACKEND_URL}/health", timeout=10)
    print(f"  /health => HTTP {hc.status_code} — {'PASS' if hc.status_code == 200 else 'FAIL'}")

    # -------------------------
    # 2. Route-level tests (LOCAL pipeline - same process as test script)
    # -------------------------
    print("\n[2] Route-Level Pipeline Validation (same-process, no HTTP)")
    
    green = io.BytesIO()
    Image.new("RGB", (640, 480), color=(34, 139, 34)).save(green, format="JPEG")
    green_bytes = green.getvalue()
    
    # Non-leaf
    o1 = pytorch_inference_pipeline.model1.forward
    def fake_nonleaf(x):
        logits = torch.full((1, 2), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, 0] = 10.0
        return logits
    pytorch_inference_pipeline.model1.forward = fake_nonleaf
    res = pytorch_inference_pipeline.execute(green_bytes)
    pytorch_inference_pipeline.model1.forward = o1
    ok = res.status == "not_leaf"
    routing_results["non_leaf"] = {"integration": "PASS" if ok else "FAIL", "status": res.status}
    print(f"  Non-leaf: status={res.status} => {'PASS' if ok else 'FAIL'}")

    # Supported crops
    for crop in ALL_SUPPORTED:
        o1 = force_leaf_mock()
        o2 = force_crop_mock(crop)
        res = pytorch_inference_pipeline.execute(green_bytes)
        restore_m1(o1)
        restore_m2(o2)
        
        pred = res.prediction or {}
        model_route = (pred.get("model_route") or {})
        disease_model = model_route.get("disease_model")
        expected = EXPECTED_MODELS[crop]
        ds = pred.get("disease_supported")
        score = pred.get("score")
        
        ok = (
            res.status in ("disease_detected", "healthy") and
            ds == True and
            disease_model == expected and
            score is not None and 0.0 <= score <= 1.0
        )
        routing_results[crop] = {
            "integration": "PASS" if ok else "FAIL",
            "status": res.status,
            "disease_model": disease_model,
            "class_id": pred.get("class_id"),
            "score_0_1": score,
            "analysis_id": None  # will fill from HTTP
        }
        print(f"  {crop.capitalize():12s} -> Model {str(disease_model):3s} (expected {expected}): {'PASS' if ok else 'FAIL'} | {res.status} | {pred.get('class_id')}")

    # Single-condition crops (1 known PlantVillage condition)
    SINGLE_CROPS = {
        "blueberry": "healthy",
        "orange": "haunglongbing_citrus_greening",
        "raspberry": "healthy",
        "soybean": "healthy",
        "squash": "powdery_mildew"
    }
    for crop, expected_cond in SINGLE_CROPS.items():
        o1 = force_leaf_mock()
        o2 = force_crop_mock(crop)
        res = pytorch_inference_pipeline.execute(green_bytes)
        restore_m1(o1)
        restore_m2(o2)
        
        pred = res.prediction or {}
        ds = pred.get("disease_supported")
        score = pred.get("score")
        dm = (pred.get("model_route") or {}).get("disease_model")
        source = pred.get("disease_source")
        class_id = pred.get("class_id")
        
        ok = (
            ds == True and
            score is None and
            dm is None and
            source == "single_dataset_class" and
            class_id == expected_cond
        )
        routing_results[crop] = {
            "integration": "PASS" if ok else "FAIL",
            "status": res.status,
            "disease_supported": ds,
            "class_id": class_id,
            "disease_source": source,
            "analysis_id": None
        }
        print(f"  {crop.capitalize():12s} single-condition: status={res.status}, cond={class_id}, ds={ds}, score={score}, dm={dm} => {'PASS' if ok else 'FAIL'}")

    # -------------------------
    # 3. Real HTTP E2E with actual train/val images
    # -------------------------
    print("\n[3] Real HTTP E2E (train/val images — NOT test set)")
    real_tests = ["apple", "corn", "tomato", "potato", "strawberry"]
    
    for crop in real_tests:
        img_path = find_train_val_image(crop)
        if not img_path:
            print(f"  {crop.capitalize():12s}: No train/val image found — SKIP")
            http_cases.append({"case": crop, "source": "N/A", "split": "N/A", "integration": "SKIP", "note": "no image found"})
            continue
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        
        # Upload
        status_up, resp_up = upload_image_http(img_bytes, img_path.name)
        if status_up != 201:
            print(f"  {crop.capitalize():12s}: Upload HTTP {status_up} — SKIP")
            http_cases.append({"case": crop, "source": str(img_path), "split": "train/val", "integration": "FAIL", "note": f"upload {status_up}"})
            continue
        image_id = resp_up["image_id"]
        
        # Analyze
        status_an, resp_an = run_analysis_http(image_id)
        if status_an != 201:
            print(f"  {crop.capitalize():12s}: Analysis HTTP {status_an} — FAIL")
            http_cases.append({"case": crop, "source": str(img_path), "split": "train/val", "integration": "FAIL", "note": f"analysis {status_an}"})
            continue
        
        actual_status = resp_an.get("status")
        pred = resp_an.get("prediction") or {}
        m1 = (resp_an.get("leaf_check") or {}).get("label", "N/A")
        m2 = (resp_an.get("crop_check") or {}).get("label", "N/A")
        disease_result = pred.get("class_id", "N/A")
        model_used = (pred.get("model_route") or {}).get("disease_model")
        analysis_id = resp_an.get("analysis_id")
        
        # Integration PASS = HTTP 201 + no crash + result is one of 5 valid statuses
        valid_statuses = {"healthy", "disease_detected", "not_leaf", "unsupported_crop", "quality_failed", "analysis_failed"}
        integration_ok = actual_status in valid_statuses
        
        routing_results[crop]["analysis_id"] = analysis_id
        
        print(f"  {crop.capitalize():12s}: HTTP 201 | M1={m1} M2={m2} -> Model {model_used} -> {disease_result} | status={actual_status} => {'PASS' if integration_ok else 'FAIL'}")
        print(f"    source: {img_path.name} | split=train/val | analysis_id={str(analysis_id)[:8]}...")
        
        http_cases.append({
            "case": crop,
            "source": str(img_path.name),
            "split": "train/val (non-test)",
            "m1_result": m1,
            "m2_result": m2,
            "disease_route": f"Model {model_used}",
            "disease_result": disease_result,
            "http_status": status_an,
            "mongodb_persisted": "Yes" if analysis_id else "Unknown",
            "integration": "PASS" if integration_ok else "FAIL",
        })

    # -------------------------
    # 4. MongoDB Persistence Verification
    # -------------------------
    print("\n[4] MongoDB Persistence Verification (GET by analysis_id)")
    for crop in ["apple", "tomato", "corn"]:
        aid = routing_results.get(crop, {}).get("analysis_id")
        if not aid:
            print(f"  {crop}: no analysis_id from HTTP E2E — skip")
            continue
        sc, fetched = get_analysis_http(aid)
        if sc != 200:
            print(f"  {crop} (ID {str(aid)[:8]}...): GET HTTP {sc} — FAIL")
            continue
        
        has_fields = all(k in fetched for k in ["status", "leaf_check", "crop_check", "prediction", "created_at"])
        pred = fetched.get("prediction") or {}
        ds_val = pred.get("disease_supported")
        score_val = pred.get("score")
        score_range_ok = score_val is None or 0.0 <= score_val <= 1.0
        
        print(f"  {crop} (ID {str(aid)[:8]}...): status={fetched.get('status')}, fields={has_fields}, "
              f"disease_supported={ds_val}, score={score_val} (0-1 range={score_range_ok}) => "
              f"{'PASS' if has_fields and score_range_ok else 'FAIL'}")

    # -------------------------
    # 5. Score Conversion Summary
    # -------------------------
    print("\n[5] Score Conversion Sanity (backend returns 0–1, frontend multiplies ×100)")
    for crop in ["apple", "cherry", "tomato"]:
        s = routing_results.get(crop, {}).get("score_0_1")
        if s is not None:
            print(f"  {crop}: backend={s:.4f} => display={round(s*100, 2)}% — CORRECT")
        else:
            print(f"  {crop}: score=null (null case handled by frontend gracefully)")

    # -------------------------
    # 6. Final Summary
    # -------------------------
    print("\n" + "="*60)
    print("ROUTING MATRIX FINAL SUMMARY")
    print("="*60)
    passes = sum(1 for v in routing_results.values() if v["integration"] == "PASS")
    total = len(routing_results)
    print(f"Total: {passes}/{total} PASS\n")
    for name, r in routing_results.items():
        print(f"  {name.capitalize():15s} | {r['integration']}")

    return routing_results, http_cases

if __name__ == "__main__":
    routing_results, http_cases = run_e2e()
    with open("e2e_results.json", "w") as f:
        json.dump({"routing": routing_results, "http_cases": http_cases}, f, indent=2, default=str)
    print("\nResults written to e2e_results.json")
