"""
Smoke test for final_refine_apple_corn.py
All temp checkpoints are deleted at the end. No real checkpoints are written.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.generic_disease_dataset import GenericDiseaseDataset
from src.models.model1_resnet9 import ResNet9Model1
from final_refine_apple_corn import (
    WORKSPACE, CONFIG_FILE, FINAL_LR, WEIGHT_DECAY,
    REFINE_MODELS, load_config, verify_source, atomic_save, cleanup_cuda
)

R = {}

def check(label, passed):
    R[label] = "PASS" if passed else "FAIL"
    print(f"  [{R[label]}] {label}")


def main():
    print("=" * 60)
    print("SMOKE TEST: final_refine_apple_corn")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    rm  = REFINE_MODELS[0]           # Apple – one model is enough
    cfg = load_config(rm["cfg_name"])
    num_classes = cfg["num_classes"]

    # 1. Strict checkpoint load / guardrail
    try:
        src = verify_source(rm)
        check("Strict checkpoint load + guardrail", True)
    except Exception as e:
        check("Strict checkpoint load + guardrail", False)
        print(f"    {e}")
        return

    # 2. Model restore
    try:
        model = ResNet9Model1(3, num_classes).to(device)
        model.load_state_dict(src["model_state"])
        check("Model state restore", True)
    except Exception as e:
        check("Model state restore", False); return

    # 3. Optimizer + LR override
    try:
        optimizer = optim.AdamW(model.parameters(), lr=FINAL_LR, weight_decay=WEIGHT_DECAY)
        if "optimizer_state" in src:
            optimizer.load_state_dict(src["optimizer_state"])
            for pg in optimizer.param_groups:
                pg["lr"] = FINAL_LR
        actual_lr = optimizer.param_groups[0]["lr"]
        check("Optimizer restore", True)
        check("LR override to 1e-4", abs(actual_lr - FINAL_LR) < 1e-12)
    except Exception as e:
        check("Optimizer restore", False)
        check("LR override to 1e-4", False)

    # 4. Scaler
    try:
        scaler = torch.amp.GradScaler("cuda")
        if "scaler_state" in src:
            scaler.load_state_dict(src["scaler_state"])
        check("Scaler restore", True)
    except Exception as e:
        check("Scaler restore", False)

    # 5. Scheduler
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5)
        check("Scheduler creation", True)
    except Exception as e:
        check("Scheduler creation", False)

    # 6. DataLoader
    try:
        train_ds = GenericDiseaseDataset(cfg["train_csv"], True,
                                         cfg["normalization_mean"], cfg["normalization_std"])
        val_ds   = GenericDiseaseDataset(cfg["train_csv"], False,
                                         cfg["normalization_mean"], cfg["normalization_std"])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
        check("DataLoader creation", True)
    except Exception as e:
        check("DataLoader creation", False); return

    # 7. Forward
    nan_inf = False
    try:
        cw = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        model.train()
        imgs, labels = next(iter(train_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        nan_inf = torch.isnan(loss).item() or torch.isinf(loss).item()
        check("Forward pass", logits.shape == (imgs.shape[0], num_classes))
    except Exception as e:
        check("Forward pass", False)

    # 8. Backward + AMP
    try:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        check("Backward + AMP", True)
    except Exception as e:
        check("Backward + AMP", False)

    # 9. Validation metrics
    m_f1 = 0.0
    try:
        model.eval()
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for imgs_v, labels_v in val_loader:
                imgs_v, labels_v = imgs_v.to(device), labels_v.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    out = model(imgs_v)
                _, pred = out.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels_v.cpu().numpy())
                break
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_classes)), zero_division=0)
        m_f1 = f1.mean()
        check("Validation metrics", True)
    except Exception as e:
        check("Validation metrics", False)

    # 10. Checkpoint save + reload (resume)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = {
                "source_checkpoint"       : str(rm["source_ckpt"]),
                "source_prev_macro_f1"    : rm["prev_f1"],
                "source_prev_val_acc"     : rm["prev_acc"],
                "final_refine_epoch"      : 1,
                "model_state"             : {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state"         : optimizer.state_dict(),
                "scaler_state"            : scaler.state_dict(),
                "scheduler_state"         : scheduler.state_dict(),
                "current_lr"              : FINAL_LR,
                "best_final_macro_f1"     : float(m_f1),
                "best_final_epoch"        : 1,
                "early_stopping_counter"  : 0,
                "validation_accuracy"     : 90.0,
                "validation_macro_f1"     : float(m_f1),
                "validation_weighted_f1"  : float(m_f1),
                "num_classes"             : num_classes,
                "class_mapping"           : cfg["class_mapping"],
                "label_to_index"          : cfg["label_to_index"],
                "normalization"           : {"mean": cfg["normalization_mean"],
                                             "std" : cfg["normalization_std"]},
                "class_weights"           : cfg["class_weights"],
                "architecture"            : cfg["architecture"],
                "image_size"              : cfg["image_size"],
            }
            ckpt_path = tmp_path / "last_final_refine_checkpoint.pth"
            atomic_save(ckpt, ckpt_path)
            check("Checkpoint save", ckpt_path.exists())

            loaded = torch.load(str(ckpt_path), map_location="cpu")
            new_model = ResNet9Model1(3, num_classes).to(device)
            new_model.load_state_dict(loaded["model_state"])
            new_opt   = optim.AdamW(new_model.parameters(), lr=FINAL_LR)
            new_opt.load_state_dict(loaded["optimizer_state"])
            new_scaler = torch.amp.GradScaler("cuda")
            new_scaler.load_state_dict(loaded["scaler_state"])
            check("Checkpoint load / resume",
                  loaded["final_refine_epoch"] == 1 and
                  abs(loaded["best_final_macro_f1"] - float(m_f1)) < 1e-6)
            del new_model, new_opt, new_scaler
    except Exception as e:
        check("Checkpoint save", False)
        check("Checkpoint load / resume", False)

    check("NaN/Inf in loss", not nan_inf)

    # 11. CUDA cleanup
    try:
        cleanup_cuda(model, optimizer, scaler)
        check("CUDA cleanup", True)
    except Exception as e:
        check("CUDA cleanup", False)

    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    overall = True
    for k, v in R.items():
        print(f"  {v}  {k}")
        if v != "PASS":
            overall = False
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
