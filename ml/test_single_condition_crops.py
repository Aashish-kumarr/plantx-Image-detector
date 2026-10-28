import sys
import os
import torch
import json
from pathlib import Path

backend_path = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\Plant_disease_detection\backend")
sys.path.append(str(backend_path))

from app.ml.pytorch_inference import pytorch_inference_pipeline, MODEL2_LABELS, SINGLE_CONDITION_CROPS
from app.schemas.analysis import AnalysisResponse

def mock_predict_crop(crop_name: str):
    idx = [k for k, v in MODEL2_LABELS.items() if v == crop_name][0]
    original_m2 = pytorch_inference_pipeline.model2.forward
    
    def fake_m2(x):
        logits = torch.full((1, 14), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, idx] = 10.0
        return logits
        
    pytorch_inference_pipeline.model2.forward = fake_m2
    
    original_m1 = pytorch_inference_pipeline.model1.forward
    def fake_m1(x):
        logits = torch.full((1, 2), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, 1] = 10.0
        return logits
        
    pytorch_inference_pipeline.model1.forward = fake_m1
    
    return original_m1, original_m2

def restore_mock(orig_m1, orig_m2):
    pytorch_inference_pipeline.model1.forward = orig_m1
    pytorch_inference_pipeline.model2.forward = orig_m2

def create_fake_image() -> bytes:
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color = 'green')
    b = io.BytesIO()
    img.save(b, format='JPEG')
    return b.getvalue()

def test_all():
    fake_bytes = create_fake_image()
    
    print("\n--- SINGLE-CONDITION CROPS TEST ---")
    single_crops = ["blueberry", "orange", "raspberry", "soybean", "squash"]
    
    results = {}
    for crop in single_crops:
        o1, o2 = mock_predict_crop(crop)
        res = pytorch_inference_pipeline.execute(fake_bytes)
        restore_mock(o1, o2)
        
        pred = res.prediction
        
        pass_cond = (
            pred["disease_supported"] is True and
            pred["score"] is None and
            pred["model_route"]["disease_model"] is None and
            pred["disease_source"] == "single_dataset_class" and
            pred["class_id"] == SINGLE_CONDITION_CROPS[crop]
        )
        
        results[crop] = {
            "status": res.status,
            "condition": pred["class_id"],
            "display_name": pred["display_name"],
            "disease_supported": pred["disease_supported"],
            "disease_score": pred["score"],
            "disease_model": pred["model_route"]["disease_model"],
            "disease_source": pred["disease_source"],
            "pass": pass_cond
        }
        
        print(f"Crop: {crop.capitalize()}")
        print(f"  Status: {res.status}")
        print(f"  Condition: {pred['class_id']} ({pred['display_name']})")
        print(f"  disease_supported: {pred['disease_supported']}")
        print(f"  disease_score: {pred['score']}")
        print(f"  disease_model: {pred['model_route']['disease_model']}")
        print(f"  disease_source: {pred['disease_source']}")
        print(f"  Result: {'PASS' if pass_cond else 'FAIL'}\n")

    print("\n--- SCHEMA SERIALIZATION CHECK ---")
    from datetime import datetime
    for crop in single_crops:
        o1, o2 = mock_predict_crop(crop)
        res = pytorch_inference_pipeline.execute(fake_bytes)
        restore_mock(o1, o2)
        
        doc = {
            "analysis_id": "test_id_123",
            "user_id": "user_123",
            "status": res.status,
            "image": {
                "image_id": "img123", "storage_key": "fake", "filename": "fake.jpg", 
                "mime_type": "image/jpeg", "size_bytes": 100, "width": 224, "height": 224
            },
            "quality": {"status": "passed", "checks": {}},
            "leaf_check": {"label": res.leaf_check.label, "score": res.leaf_check.score, "model_version": "v1.0"},
            "crop_check": {"label": res.crop_check.label, "score": res.crop_check.score, "model_version": "v1.0"},
            "prediction": res.prediction,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        AnalysisResponse(**doc)
        print(f"Serialization for {crop.capitalize()}: PASS")

if __name__ == "__main__":
    test_all()
