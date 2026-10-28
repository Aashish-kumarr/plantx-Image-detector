import os
import json
import torch
from pathlib import Path

def main():
    workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
    manifest_path = workspace / "metadata" / "models5_11_final_champions.json"
    
    checkpoints = [
        (5, 'model5_apple', 'checkpoints/model5_apple_final_refine/best_final_refined_model.pth', 'FINAL_REFINE'),
        (6, 'model6_cherry', 'checkpoints/model6_cherry/best_model.pth', 'INITIAL'),
        (7, 'model7_corn', 'checkpoints/model7_corn_refine/best_refined_model.pth', 'PREVIOUS_REFINE'),
        (8, 'model8_grape', 'checkpoints/model8_grape/best_model.pth', 'INITIAL'),
        (9, 'model9_peach', 'checkpoints/model9_peach/best_model.pth', 'INITIAL'),
        (10, 'model10_pepper', 'checkpoints/model10_pepper/best_model.pth', 'INITIAL'),
        (11, 'model11_strawberry', 'checkpoints/model11_strawberry/best_model.pth', 'INITIAL'),
    ]
    
    manifest = []
    for model_number, model_name, relative_path, stage in checkpoints:
        path = workspace / relative_path
        s = torch.load(str(path), map_location='cpu')
        
        # Determine exact keys based on stage
        if stage == 'FINAL_REFINE':
            val_acc = s['validation_accuracy']
            val_f1 = s['best_final_macro_f1']
            crop = s['training_configuration']['crop']
        elif stage == 'PREVIOUS_REFINE':
            val_acc = s['validation_accuracy']
            val_f1 = s['best_refined_macro_f1']
            crop = s['training_configuration']['crop']
        else:
            val_acc = s['validation_accuracy']
            val_f1 = s['best_macro_f1']
            crop = s['crop']
            
        # Common parameters
        if stage in ['FINAL_REFINE', 'PREVIOUS_REFINE']:
            num_classes = s['num_classes']
            class_mapping = s['class_mapping']
            normalization_mean = s['normalization']['mean']
            normalization_std = s['normalization']['std']
            architecture = s['architecture']
            image_size = s['image_size']
        else:
            num_classes = s['num_classes']
            class_mapping = s['class_mapping']
            normalization_mean = s['normalization']['mean']
            normalization_std = s['normalization']['std']
            architecture = s['architecture']
            image_size = s['image_size']
        
        manifest.append({
            "model_number": model_number,
            "model_name": model_name,
            "crop": crop,
            "checkpoint_path": str(path),
            "checkpoint_stage": stage,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "num_classes": num_classes,
            "class_mapping": class_mapping,
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "architecture": architecture,
            "image_size": image_size,
            "test_status": "LOCKED_NOT_EVALUATED"
        })
        
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Created manifest with {len(manifest)} champions.")
        
if __name__ == '__main__':
    main()
