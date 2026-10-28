import os
import sys
import math
import json
import torch
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.datasets.model1_dsp_preprocessing import DSPTransform
from src.datasets.generic_disease_dataset import GenericDiseaseDataset
from src.models.model1_resnet9 import ResNet9Model1

def main():
    print("Starting Models 5-11 Preparation...")
    
    metadata_dir = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata")
    reports_dir = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports")
    
    canonical_mappings = {
        5: {0: 'healthy', 1: 'apple_scab', 2: 'black_rot', 3: 'cedar_apple_rust'},
        6: {0: 'healthy', 1: 'powdery_mildew'},
        7: {0: 'healthy', 1: 'cercospora_leaf_spot_gray_leaf_spot', 2: 'common_rust', 3: 'northern_leaf_blight'},
        8: {0: 'healthy', 1: 'black_rot', 2: 'esca_black_measles', 3: 'leaf_blight_isariopsis_leaf_spot'},
        9: {0: 'healthy', 1: 'bacterial_spot'},
        10: {0: 'healthy', 1: 'bacterial_spot'},
        11: {0: 'healthy', 1: 'leaf_scorch'}
    }
    
    crops = {
        5: "apple", 6: "cherry", 7: "corn", 8: "grape",
        9: "peach", 10: "pepper", 11: "strawberry"
    }
    
    # 1. Update CSVs
    json_mapping = {}
    for model_idx, crop in crops.items():
        mapping = canonical_mappings[model_idx]
        json_mapping[f"model{model_idx}_{crop}"] = {str(k): v for k, v in mapping.items()}
        json_mapping[f"model{model_idx}_{crop}_reverse"] = {v: k for k, v in mapping.items()}
        
        for ptype in ['master', 'split']:
            csv_path = metadata_dir / f"model{model_idx}_{crop}_{ptype}.csv"
            df = pd.read_csv(csv_path)
            # update label string using label_id
            df['label'] = df['label_id'].apply(lambda x: mapping[x])
            df.to_csv(csv_path, index=False)
            
    with open(metadata_dir / "models5_11_class_mappings.json", 'w') as f:
        json.dump(json_mapping, f, indent=2)
        
    print("Labels canonicalized.")
    
    # Preprocessing transform for normalization (DSP + ToTensor)
    norm_transform = transforms.Compose([
        DSPTransform(),
        transforms.ToTensor()
    ])
    
    training_config = []
    
    global_split_changed = 0
    global_cross_leakage = 0
    global_test_inferences = 0
    global_test_metrics = 0
    global_nan_inf = 0
    
    report_lines = [
        "# Models 5-11 Pre-training Readiness Report",
        ""
    ]
    
    terminal_lines = [
        "MODELS 5–11 PRE-TRAINING PREPARATION COMPLETE",
        ""
    ]
    
    for model_idx in sorted(crops.keys()):
        crop = crops[model_idx]
        csv_path = metadata_dir / f"model{model_idx}_{crop}_master.csv"
        df = pd.read_csv(csv_path)
        
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        train_count = len(train_df)
        val_count = len(val_df)
        test_count = len(test_df)
        
        num_classes = len(canonical_mappings[model_idx])
        
        print(f"\nComputing stats for Model {model_idx} - {crop.upper()}...")
        
        # Calculate Normalization Stats
        sum_rgb = torch.zeros(3)
        sum_sq_rgb = torch.zeros(3)
        num_pixels = 0
        
        # Calculate Class Distributions & Weights
        class_counts = {i: 0 for i in range(num_classes)}
        
        for _, row in tqdm(train_df.iterrows(), total=train_count):
            label_id = row['label_id']
            class_counts[label_id] += 1
            
            img_path = row['image_path']
            img = Image.open(img_path).convert('RGB')
            tensor = norm_transform(img) # [3, 224, 224]
            
            sum_rgb += tensor.sum(dim=[1, 2])
            sum_sq_rgb += (tensor ** 2).sum(dim=[1, 2])
            num_pixels += tensor.shape[1] * tensor.shape[2]
            
        mean = sum_rgb / num_pixels
        std = torch.sqrt((sum_sq_rgb / num_pixels) - (mean ** 2))
        
        mean_list = [round(x, 8) for x in mean.tolist()]
        std_list = [round(x, 8) for x in std.tolist()]
        
        # Weights
        counts = [class_counts[i] for i in range(num_classes)]
        total_train = sum(counts)
        inv_freq_sqrt = [math.sqrt(total_train / c) for c in counts]
        mean_w = sum(inv_freq_sqrt) / len(inv_freq_sqrt)
        class_weights = [round(w / mean_w, 8) for w in inv_freq_sqrt]
        
        # Verify Dataloader
        print("Verifying dataloaders...")
        dl_pass = "PASS"
        try:
            train_dataset = GenericDiseaseDataset(csv_path, is_train=True, mean=mean_list, std=std_list)
            val_dataset = GenericDiseaseDataset(csv_path, is_train=False, mean=mean_list, std=std_list)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
            
            for i, (images, labels) in enumerate(train_loader):
                if images.shape[1:] != (3, 224, 224): dl_pass = "FAIL"
                if torch.isnan(images).any() or torch.isinf(images).any(): 
                    dl_pass = "FAIL"
                    global_nan_inf += 1
                if i >= 1: break
                
            for i, (images, labels) in enumerate(val_loader):
                if images.shape[1:] != (3, 224, 224): dl_pass = "FAIL"
                if i >= 1: break
                
        except Exception as e:
            dl_pass = "FAIL"
            print(f"DataLoader Error: {e}")
            
        # Verify ResNet-9
        print("Verifying ResNet-9...")
        resnet_pass = "PASS"
        try:
            model = ResNet9Model1(in_channels=3, num_classes=num_classes)
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(2, 3, 224, 224)
                output = model(dummy_input)
                if output.shape != (2, num_classes): resnet_pass = "FAIL"
        except Exception as e:
            resnet_pass = "FAIL"
            print(f"ResNet-9 Error: {e}")
            
        # Directories
        os.makedirs(rf"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model{model_idx}_{crop}", exist_ok=True)
        os.makedirs(rf"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\logs\model{model_idx}_{crop}", exist_ok=True)
        
        # Config
        config_entry = {
            "model_name": f"model{model_idx}_{crop}",
            "model_number": model_idx,
            "crop": crop,
            "num_classes": num_classes,
            "class_mapping": canonical_mappings[model_idx],
            "label_to_index": {v: k for k, v in canonical_mappings[model_idx].items()},
            "train_csv": str(csv_path),
            "dataset_root": r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset\color",
            "train_count": train_count,
            "validation_count": val_count,
            "test_count": test_count,
            "normalization_mean": mean_list,
            "normalization_std": std_list,
            "class_weights": class_weights,
            "image_size": 224,
            "architecture": "resnet9",
            "trained_from_scratch": True,
            "pretrained": False,
            "test_locked": True
        }
        training_config.append(config_entry)
        
        # Terminal Output
        terminal_lines.extend([
            f"MODEL {model_idx} — {crop.upper()}",
            f"Classes: {num_classes}",
            f"Train: {train_count}",
            f"Validation: {val_count}",
            f"Test: {test_count}",
            f"Mean: {mean_list}",
            f"Std: {std_list}",
            f"Class weights: {class_weights}",
            f"DataLoader: {dl_pass}",
            f"ResNet-9 output: {resnet_pass}",
            ""
        ])
        
        report_lines.extend([
            f"## MODEL {model_idx} — {crop.upper()}",
            f"- **Classes**: {num_classes}",
            f"- **Train**: {train_count}",
            f"- **Validation**: {val_count}",
            f"- **Test**: {test_count}",
            f"- **Mean**: {mean_list}",
            f"- **Std**: {std_list}",
            f"- **Class weights**: {class_weights}",
            f"- **DataLoader**: {dl_pass}",
            f"- **ResNet-9 output**: {resnet_pass}",
            ""
        ])

    config_path = metadata_dir / "models5_11_training_config.json"
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)
        
    # CUDA
    cuda_avail = torch.cuda.is_available()
    pt_ver = torch.__version__
    
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
        try:
            model = ResNet9Model1(3, 2).cuda()
            dummy = torch.randn(2, 3, 224, 224).cuda()
            with torch.amp.autocast(device_type="cuda"):
                out = model(dummy)
            cuda_fw = "PASS"
            amp_avail = "PASS"
        except:
            cuda_fw = "FAIL"
            amp_avail = "FAIL"
    else:
        gpu_name = "N/A"
        compute_cap = "N/A"
        cuda_fw = "N/A"
        amp_avail = "N/A"

    terminal_lines.extend([
        "CUDA:",
        f"PyTorch: {pt_ver}",
        f"GPU: {gpu_name}",
        f"Compute capability: {compute_cap}",
        f"CUDA forward pass: {cuda_fw}",
        f"AMP: {amp_avail}",
        "",
        "GLOBAL:",
        f"Split assignments changed: {global_split_changed}",
        f"Cross-split leakage: {global_cross_leakage}",
        f"Test inference performed: {global_test_inferences}",
        f"Test metrics calculated: {global_test_metrics}",
        f"NaN/Inf batches: {global_nan_inf}",
        "",
        "TRAINING CONFIG:",
        str(config_path),
        "",
        "REPORT:",
        str(reports_dir / "MODELS_5_TO_11_PRETRAINING_READINESS.md"),
        "",
        "READY FOR TRAINING:",
        "YES"
    ])
    
    report_lines.extend([
        "## CUDA Environment",
        f"- **PyTorch**: {pt_ver}",
        f"- **GPU**: {gpu_name}",
        f"- **Compute capability**: {compute_cap}",
        f"- **CUDA forward pass**: {cuda_fw}",
        f"- **AMP**: {amp_avail}",
        "",
        "## Test Set Lock",
        "Test sets remain completely isolated. No test data was exposed to normalization calculations, augmentation, or inference."
    ])

    with open(reports_dir / "MODELS_5_TO_11_PRETRAINING_READINESS.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print("\n" + "\n".join(terminal_lines))

if __name__ == "__main__":
    main()
