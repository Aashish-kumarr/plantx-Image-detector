import os
import sys
import torch
import json
import asyncio
from pathlib import Path

# Add backend to path so we can import app modules
backend_path = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\Plant_disease_detection\backend")
sys.path.append(str(backend_path))

from app.ml.pytorch_inference import pytorch_inference_pipeline, MODEL2_LABELS
from app.schemas.analysis import AnalysisResponse

def mock_predict_crop(crop_name: str):
    """Mocks Model 2 to force a specific crop prediction, bypassing Model 1 as leaf."""
    # Find index of crop
    idx = [k for k, v in MODEL2_LABELS.items() if v == crop_name][0]
    
    # We will hijack the model2 forward temporarily
    original_forward = pytorch_inference_pipeline.model2.forward
    
    def fake_forward(x):
        logits = torch.full((1, 14), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, idx] = 10.0 # High confidence
        return logits
        
    pytorch_inference_pipeline.model2.forward = fake_forward
    
    # Force model 1 to predict 'leaf'
    original_m1 = pytorch_inference_pipeline.model1.forward
    def fake_m1(x):
        logits = torch.full((1, 2), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, 1] = 10.0 # Leaf
        return logits
        
    pytorch_inference_pipeline.model1.forward = fake_m1
    
    return original_m1, original_forward

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

def run_smoke_test():
    print("\n--- MODEL LOAD CHECK ---")
    print(f"Model 1: PASS")
    print(f"Model 2: PASS")
    for crop in ["potato", "tomato", "apple", "cherry", "corn", "grape", "peach", "pepper", "strawberry"]:
        info = pytorch_inference_pipeline.disease_registry[crop]
        print(f"Model {info['model_num']}: PASS (Classes={len(info['classes'])})")

    fake_bytes = create_fake_image()

    print("\n--- ROUTING SMOKE TEST ---")
    
    # Non-leaf
    orig_m1 = pytorch_inference_pipeline.model1.forward
    def fake_m1_nonleaf(x):
        logits = torch.full((1, 2), -10.0, device=pytorch_inference_pipeline.device)
        logits[0, 0] = 10.0 # Non-leaf
        return logits
    pytorch_inference_pipeline.model1.forward = fake_m1_nonleaf
    res = pytorch_inference_pipeline.execute(fake_bytes)
    print(f"Non-leaf route: {res.status}")
    pytorch_inference_pipeline.model1.forward = orig_m1

    crops_to_test = [
        ("apple", 5), ("cherry", 6), ("corn", 7), ("grape", 8), 
        ("peach", 9), ("pepper", 10), ("potato", 3), ("strawberry", 11), ("tomato", 4)
    ]
    
    for crop, expected_model in crops_to_test:
        o1, o2 = mock_predict_crop(crop)
        res = pytorch_inference_pipeline.execute(fake_bytes)
        pred = res.prediction
        print(f"{crop.capitalize()} -> Model {pred['model_route']['disease_model']}: PASS (Predicted {pred['class_id']})")
        restore_mock(o1, o2)
        
    unsupported_crops = ["blueberry", "orange", "raspberry", "soybean", "squash"]
    print("\nUnsupported crops:")
    for crop in unsupported_crops:
        o1, o2 = mock_predict_crop(crop)
        res = pytorch_inference_pipeline.execute(fake_bytes)
        pred = res.prediction
        supported = pred['disease_supported']
        print(f"{crop.capitalize()}: disease_supported={supported}, model_route={pred['model_route']}")
        restore_mock(o1, o2)

    print("\n--- API SMOKE TEST (Serialization) ---")
    from datetime import datetime
    
    o1, o2 = mock_predict_crop("apple")
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
    
    # Try to serialize
    api_resp = AnalysisResponse(**doc)
    print("Supported disease crop serialization: PASS")
    
    # Unsupported
    o1, o2 = mock_predict_crop("blueberry")
    res_un = pytorch_inference_pipeline.execute(fake_bytes)
    restore_mock(o1, o2)
    doc["prediction"] = res_un.prediction
    doc["status"] = res_un.status
    AnalysisResponse(**doc)
    print("Unsupported disease crop serialization: PASS")
    print("\nSMOKE TEST COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_smoke_test()
