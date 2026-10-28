
import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
sys.path.insert(0, BASE_DIR)

from src.models.model2_14class_resnet9 import create_model2_14class_resnet9
from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing
from src.datasets.model2_transforms import get_model2_train_transform, get_model2_eval_transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Path-remapping dataset
# ---------------------------------------------------------------------------
OLD_PATH_PREFIX = r"C:\Users\ak021\Desktop\plantDx\plant image detector"
OLD_PATH_PREFIX2 = r"c:\Users\ak021\Desktop\plant image detector"
NEW_PATH_PREFIX = BASE_DIR

def remap_image_path(raw_path: str) -> str:
    """Replace old base prefix with current BASE_DIR."""
    if raw_path.lower().startswith(OLD_PATH_PREFIX.lower()):
        return NEW_PATH_PREFIX + raw_path[len(OLD_PATH_PREFIX):]
    if raw_path.lower().startswith(OLD_PATH_PREFIX2.lower()):
        return NEW_PATH_PREFIX + raw_path[len(OLD_PATH_PREFIX2):]
    return raw_path

class PathRemappedModel2Dataset(Dataset):
    def __init__(self, csv_path: str, split_name: str, transform):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split_name].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = remap_image_path(str(row["image_path"]))
        label = int(row["label_id"])
        pil_img = Image.open(img_path).convert("RGB")
        dsp_pil = apply_dsp_preprocessing(pil_img)
        return self.transform(dsp_pil), label

def build_model2_dataloaders(csv_path: str, batch_size: int, num_workers: int):
    train_ds = PathRemappedModel2Dataset(csv_path, "train", get_model2_train_transform())
    val_ds   = PathRemappedModel2Dataset(csv_path, "val",   get_model2_eval_transform())
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIGINAL_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "model2_14class", "best_model2_14class.pth")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "model2_14class_refine")
RESULTS_DIR    = os.path.join(BASE_DIR, "results",     "model2_14class_refine")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports",     "model2_14class_refine")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

HISTORY_FILE          = os.path.join(RESULTS_DIR, "training_history.csv")
CONFUSION_MATRIX_FILE = os.path.join(RESULTS_DIR, "confusion_matrix.png")
REPORT_FILE           = os.path.join(REPORTS_DIR, "model2_refinement_report.txt")

BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_model2_14class_refine.pth")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last_model2_14class_refine.pth")

MAX_REFINE_EPOCHS  = 5
REFINE_LR          = 2e-4
WEIGHT_DECAY       = 1e-4
BATCH_SIZE         = 64
NUM_WORKERS        = 0
PATIENCE           = 3

# We require the loaded checkpoint to match these to ensure we branch from the right state
EXPECTED_EPOCH = 3
EXPECTED_F1    = 0.8223

# ---------------------------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------------------------
LABEL_MAP = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
    4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight',
    21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

CROP_LABEL_MAP = {
    0: 'apple', 1: 'blueberry', 2: 'cherry', 3: 'corn', 4: 'grape',
    5: 'orange', 6: 'peach', 7: 'pepper', 8: 'potato', 9: 'raspberry',
    10: 'soybean', 11: 'squash', 12: 'strawberry', 13: 'tomato'
}

def get_class_weights():
    # Existing square-root inverse-frequency weights
    return torch.tensor([
        0.8399, 1.2224, 1.0980, 0.7703, 0.7421, 0.6420, 0.9224, 0.9540,
        1.0271, 2.4479, 0.6691, 1.1054, 1.2047, 0.3546
    ], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Refine Epoch {epoch}/{MAX_REFINE_EPOCHS} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += images.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc  = (correct / total) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Refine Epoch {epoch}/{MAX_REFINE_EPOCHS} [val]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += images.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = (correct / total) * 100.0
    
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(14), zero_division=0)
    
    return epoch_loss, epoch_acc, prec, rec, f1, pc_prec, pc_rec, pc_f1, all_labels, all_preds

def write_history(epoch, train_loss, train_acc, val_loss, val_acc, val_prec, val_rec, val_f1, lr, t_epoch, pc_prec, pc_rec, pc_f1):
    file_exists = os.path.exists(HISTORY_FILE)
    
    headers = [
        "refine_epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "val_macro_precision", "val_macro_recall", "val_macro_f1", "learning_rate", "epoch_time"
    ]
    for i in range(14):
        c_name = CROP_LABEL_MAP[i]
        headers.extend([f"{c_name}_precision", f"{c_name}_recall", f"{c_name}_f1"])
        
    with open(HISTORY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
            
        row = [epoch, train_loss, train_acc, val_loss, val_acc, val_prec, val_rec, val_f1, lr, t_epoch]
        for i in range(14):
            row.extend([pc_prec[i], pc_rec[i], pc_f1[i]])
        writer.writerow(row)

def generate_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds, labels=range(14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[CROP_LABEL_MAP[i] for i in range(14)])
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Model 2 - 14-Class Crop Classifier (Refinement Best)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    
    confusions = []
    for i in range(14):
        for j in range(14):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": CROP_LABEL_MAP[i],
                    "pred": CROP_LABEL_MAP[j],
                    "count": cm[i, j]
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions

def write_report(best_f1, best_acc, best_epoch, original_f1, original_acc, pc_prec, pc_rec, pc_f1, confusions):
    potato_idx, tomato_idx, rasp_idx = 8, 13, 9
    
    lines = [
        "Model 2 Refinement Report",
        "=========================",
        "",
        "--- Baseline (Original Epoch 3) ---",
        f"Val Accuracy : {original_acc:.2f}%",
        f"Val Macro-F1 : {original_f1:.4f}",
        "",
        "--- Best Refinement Result ---",
        f"Best Refine Epoch : {best_epoch} / {MAX_REFINE_EPOCHS}",
        f"Val Accuracy      : {best_acc:.2f}%",
        f"Val Macro-F1      : {best_f1:.4f}",
        "",
        "--- Key Crop Metrics (Best Refinement) ---",
        f"Potato    -> Prec: {pc_prec[potato_idx]:.4f} | Rec: {pc_rec[potato_idx]:.4f} | F1: {pc_f1[potato_idx]:.4f}",
        f"Tomato    -> Prec: {pc_prec[tomato_idx]:.4f} | Rec: {pc_rec[tomato_idx]:.4f} | F1: {pc_f1[tomato_idx]:.4f}",
        f"Raspberry -> Prec: {pc_prec[rasp_idx]:.4f} | Rec: {pc_rec[rasp_idx]:.4f} | F1: {pc_f1[rasp_idx]:.4f}",
        "",
        "--- Comparison ---",
        "Is Refinement better? " + ("YES" if best_f1 > original_f1 else "NO (Failed to beat baseline)"),
        "",
        "--- Top 10 Confusions ---"
    ]
    
    for c in confusions[:10]:
        lines.append(f"True: {c['true']:<12} | Pred: {c['pred']:<12} | Count: {c['count']}")
        
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))


def run_resume():
    print("="*68)
    print("  Model 2 - 14-Class Crop Classifier  |  Refinement Training")
    print("="*68)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    
    if not os.path.exists(ORIGINAL_CHECKPOINT):
        print(f"ERROR: Original checkpoint not found at {ORIGINAL_CHECKPOINT}")
        sys.exit(1)
        
    print(f"Loading checkpoint: {ORIGINAL_CHECKPOINT}")
    ckpt = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    
    ckpt_epoch = ckpt.get("epoch", -1)
    ckpt_f1 = ckpt.get("best_macro_f1", 0.0)
    if ckpt_f1 == 0.0: ckpt_f1 = ckpt.get("best_f1", 0.0)
        
    print(f"  Checkpoint Epoch : {ckpt_epoch}")
    print(f"  Checkpoint F1    : {ckpt_f1:.4f}")
    
    if ckpt_epoch != EXPECTED_EPOCH:
        print(f"ERROR: Checkpoint is from epoch {ckpt_epoch}, expected {EXPECTED_EPOCH}.")
        sys.exit(1)
        
    if abs(ckpt_f1 - EXPECTED_F1) > 0.001:
        print(f"ERROR: Checkpoint best F1 is {ckpt_f1:.4f}, expected {EXPECTED_F1}.")
        sys.exit(1)
        
    print("  Guard-rails OK.")
    
    model = create_model2_14class_resnet9().to(device)
    model.load_state_dict(ckpt["model_state"])
    
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=REFINE_LR, weight_decay=WEIGHT_DECAY)
    if "optimizer_state" in ckpt:
        print("  Restoring optimizer state from checkpoint...")
        optimizer.load_state_dict(ckpt["optimizer_state"])
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = REFINE_LR
    print(f"  Optimizer LR forced to {REFINE_LR}")
    
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6
    )
    
    CSV_PATH = os.path.join(BASE_DIR, "metadata", "model2_14class_split.csv")
    print(f"\nBuilding dataloaders from {CSV_PATH} ...")
    train_loader, val_loader = build_model2_dataloaders(
        csv_path=CSV_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )
    
    best_f1 = EXPECTED_F1
    best_acc = 86.74
    best_epoch = 0
    patience_counter = 0
    best_pc_prec, best_pc_rec, best_pc_f1 = None, None, None
    best_labels, best_preds = None, None
    
    for epoch in range(1, MAX_REFINE_EPOCHS + 1):
        print(f"\n==================================================")
        print(f"  REFINE EPOCH {epoch}/{MAX_REFINE_EPOCHS}")
        print(f"==================================================")
        
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_prec, val_rec, val_f1, pc_prec, pc_rec, pc_f1, all_labels, all_preds = validate_epoch(model, val_loader, criterion, device, epoch)
        epoch_time = time.time() - start_time
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTime: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")
        print(f"Val Macro-F1: {val_f1:.4f}")
        
        print(f"  Potato    -> Prec: {pc_prec[8]:.4f} | Rec: {pc_rec[8]:.4f} | F1: {pc_f1[8]:.4f}")
        print(f"  Tomato    -> Prec: {pc_prec[13]:.4f} | Rec: {pc_rec[13]:.4f} | F1: {pc_f1[13]:.4f}")
        print(f"  Raspberry -> Prec: {pc_prec[9]:.4f} | Rec: {pc_rec[9]:.4f} | F1: {pc_f1[9]:.4f}")
        
        write_history(epoch, train_loss, train_acc, val_loss, val_acc, val_prec, val_rec, val_f1, current_lr, epoch_time, pc_prec, pc_rec, pc_f1)
        
        scheduler.step(val_loss)
        
        checkpoint = {
            'refine_epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'val_macro_f1': val_f1,
            'architecture': ckpt.get("architecture"),
            'normalization': ckpt.get("normalization"),
            'class_weights': ckpt.get("class_weights"),
            'label_mapping': ckpt.get("label_mapping")
        }
        torch.save(checkpoint, LAST_CKPT_PATH)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_pc_prec, best_pc_rec, best_pc_f1 = pc_prec, pc_rec, pc_f1
            best_labels, best_preds = all_labels, all_preds
            
            torch.save(checkpoint, BEST_CKPT_PATH)
            print(f"*** NEW BEST REFINEMENT MODEL (F1 {best_f1:.4f}) SAVED ***")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered!")
            break
            
    print("\nRefinement finished.")
    if best_epoch > 0:
        print(f"Generating confusion matrix for best refine epoch {best_epoch}...")
        confusions = generate_confusion_matrix(best_labels, best_preds)
        write_report(best_f1, best_acc, best_epoch, EXPECTED_F1, 86.74, best_pc_prec, best_pc_rec, best_pc_f1, confusions)
        print(f"Report written to {REPORT_FILE}")
    else:
        print("Failed to beat the original baseline (0.8223). No best refinement checkpoint saved.")

if __name__ == "__main__":
    run_resume()
