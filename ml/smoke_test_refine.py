"""
Smoke test for refine_apple_corn.py
Tests all components using temporary files only. No real checkpoints written.
"""

import sys
import json
import tempfile
import shutil
import gc
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
from refine_apple_corn import (
    WORKSPACE, CONFIG_FILE, REFINE_LR, WEIGHT_DECAY,
    load_config, verify_source_checkpoint, atomic_save, cleanup_cuda
)

RESULTS = {}

def check(label, passed):
    RESULTS[label] = "PASS" if passed else "FAIL"
    print(f"  [{RESULTS[label]}] {label}")

def main():
    print("=" * 60)
    print("SMOKE TEST: refine_apple_corn")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    with open(str(CONFIG_FILE)) as f:
        configs = json.load(f)
    cfg = next(c for c in configs if c["model_name"] == "model5_apple")
    num_classes = cfg["num_classes"]

    # ── 1. Strict checkpoint load ──────────────────────────────────────────
    rm = {
        "model_name": "model5_apple",
        "source_checkpoint": WORKSPACE / "checkpoints" / "model5_apple" / "best_model.pth",
        "expected_epoch": 2, "expected_f1": 0.7583,
        "expected_acc": 81.01, "expected_classes": 4,
    }
    try:
        src_state = verify_source_checkpoint(rm)
        check("Checkpoint strict load", True)
    except Exception as e:
        check("Checkpoint strict load", False)
        print(f"    Error: {e}")
        return

    # ── 2. Model restore ───────────────────────────────────────────────────
    try:
        model = ResNet9Model1(in_channels=3, num_classes=num_classes).to(device)
        model.load_state_dict(src_state["model_state"])
        check("Model state restore", True)
    except Exception as e:
        check("Model state restore", False)
        return

    # ── 3. Optimizer restore + LR override ────────────────────────────────
    try:
        optimizer = optim.AdamW(model.parameters(), lr=REFINE_LR, weight_decay=WEIGHT_DECAY)
        if "optimizer_state" in src_state:
            optimizer.load_state_dict(src_state["optimizer_state"])
            for pg in optimizer.param_groups:
                pg["lr"] = REFINE_LR
        actual_lr = optimizer.param_groups[0]["lr"]
        check("Optimizer restore", True)
        check("LR override to 2e-4", abs(actual_lr - REFINE_LR) < 1e-10)
    except Exception as e:
        check("Optimizer restore", False)
        check("LR override to 2e-4", False)

    # ── 4. Scaler restore ─────────────────────────────────────────────────
    try:
        scaler = torch.amp.GradScaler("cuda")
        if "scaler_state" in src_state:
            scaler.load_state_dict(src_state["scaler_state"])
        check("Scaler restore", True)
    except Exception as e:
        check("Scaler restore", False)

    # ── 5. Scheduler ──────────────────────────────────────────────────────
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5)
        check("Scheduler creation", True)
    except Exception as e:
        check("Scheduler creation", False)

    # ── 6. DataLoader (1 batch) ───────────────────────────────────────────
    try:
        train_ds = GenericDiseaseDataset(cfg["train_csv"], True,  cfg["normalization_mean"], cfg["normalization_std"])
        val_ds   = GenericDiseaseDataset(cfg["train_csv"], False, cfg["normalization_mean"], cfg["normalization_std"])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
        check("DataLoader creation", True)
    except Exception as e:
        check("DataLoader creation", False)
        return

    # ── 7. Forward ─────────────────────────────────────────────────────────
    nan_detected = False
    try:
        cw = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)

        model.train()
        imgs, labels = next(iter(train_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        check("Forward pass", logits.shape == (imgs.shape[0], num_classes))
    except Exception as e:
        check("Forward pass", False)

    # ── 8. Backward + optimizer + AMP ────────────────────────────────────
    try:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        nan_detected = torch.isnan(loss).item() or torch.isinf(loss).item()
        check("Backward pass", True)
        check("Optimizer step (AMP)", True)
    except Exception as e:
        check("Backward pass", False)
        check("Optimizer step (AMP)", False)

    # ── 9. Validation metrics ─────────────────────────────────────────────
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
                break   # one batch only
        prec, rec, f1, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_classes)), zero_division=0)
        m_f1 = f1.mean()
        check("Validation metrics", True)
    except Exception as e:
        check("Validation metrics", False)

    # ── 10. Checkpoint save / load ────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state = {
                "source_checkpoint": str(rm["source_checkpoint"]),
                "source_epoch": 2,
                "source_macro_f1": 0.7583,
                "refine_epoch": 1,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_refined_macro_f1": float(m_f1),
                "best_refine_epoch": 1,
                "early_stopping_counter": 0,
                "validation_accuracy": 80.0,
                "validation_macro_f1": float(m_f1),
                "validation_weighted_f1": float(m_f1),
                "class_mapping": cfg["class_mapping"],
                "label_to_index": cfg["label_to_index"],
                "normalization": {"mean": cfg["normalization_mean"], "std": cfg["normalization_std"]},
                "class_weights": cfg["class_weights"],
                "architecture": cfg["architecture"],
                "num_classes": num_classes,
                "image_size": cfg["image_size"],
                "current_lr": REFINE_LR,
            }
            ckpt_path = tmp_path / "last_refine_checkpoint.pth"
            atomic_save(state, ckpt_path)
            check("Checkpoint save", ckpt_path.exists())

            # Resume check
            loaded = torch.load(str(ckpt_path), map_location="cpu")
            resume_model = ResNet9Model1(3, num_classes).to(device)
            resume_model.load_state_dict(loaded["model_state"])
            resume_opt   = optim.AdamW(resume_model.parameters(), lr=REFINE_LR)
            resume_opt.load_state_dict(loaded["optimizer_state"])
            resume_scaler = torch.amp.GradScaler("cuda")
            resume_scaler.load_state_dict(loaded["scaler_state"])
            check("Checkpoint load / resume", loaded["refine_epoch"] == 1)
            check("NaN/Inf in loss", not nan_detected)

            del resume_model, resume_opt, resume_scaler
    except Exception as e:
        check("Checkpoint save", False)
        check("Checkpoint load / resume", False)
        check("NaN/Inf in loss", False)

    # ── 11. CUDA cleanup ──────────────────────────────────────────────────
    try:
        cleanup_cuda(model, optimizer, scaler)
        check("CUDA cleanup", True)
    except Exception as e:
        check("CUDA cleanup", False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    all_pass = True
    for k, v in RESULTS.items():
        print(f"  {v}  {k}")
        if v != "PASS":
            all_pass = False

    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
