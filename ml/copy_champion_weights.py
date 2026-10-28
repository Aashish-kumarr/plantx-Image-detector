import os
import shutil
import hashlib
import json
import torch
from pathlib import Path

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    ml_workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
    backend_workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\Plant_disease_detection\backend")
    weights_dir = backend_workspace / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Models 5-11 mapping
    models_mapping = [
        (5, "apple", r"checkpoints\model5_apple_final_refine\best_final_refined_model.pth"),
        (6, "cherry", r"checkpoints\model6_cherry\best_model.pth"),
        (7, "corn", r"checkpoints\model7_corn_refine\best_refined_model.pth"),
        (8, "grape", r"checkpoints\model8_grape\best_model.pth"),
        (9, "peach", r"checkpoints\model9_peach\best_model.pth"),
        (10, "pepper", r"checkpoints\model10_pepper\best_model.pth"),
        (11, "strawberry", r"checkpoints\model11_strawberry\best_model.pth"),
    ]
    
    results = []
    
    for model_num, crop, src_rel in models_mapping:
        src = ml_workspace / src_rel
        dest = weights_dir / f"model{model_num}_{crop}.pth"
        
        # Copy
        shutil.copy2(src, dest)
        
        # Verify
        sha_src = compute_sha256(src)
        sha_dest = compute_sha256(dest)
        
        match = (sha_src == sha_dest)
        print(f"Model {model_num} {crop.capitalize()}:")
        print(f"Source: {src}")
        print(f"Destination: {dest}")
        print(f"SHA-256 match: {match} ({sha_dest})\n")
        
        results.append({
            "model_num": model_num,
            "crop": crop,
            "dest": str(dest),
            "sha256": sha_dest,
            "src": str(src)
        })

    # Note: the prompt asks to create backend\weights\models_manifest.json with all Models 1-11.
    # We will get metadata for Models 1-4 from their current weights inside app\ml\pytorch_models\weights
    old_weights_dir = backend_workspace / "app" / "ml" / "pytorch_models" / "weights"
    
    # Let's write out a basic JSON here that we will extend later with model metadata
    with open(weights_dir / "copied_models.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
