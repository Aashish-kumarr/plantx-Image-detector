import os, sys, torch, pandas as pd

BASE_DIR = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
sys.path.insert(0, BASE_DIR)

CKPT = os.path.join(BASE_DIR, "checkpoints", "model2_14class", "best_model2_14class.pth")
HIST = os.path.join(BASE_DIR, "results", "model2_14class", "training_history.csv")
PC   = os.path.join(BASE_DIR, "results", "model2_14class", "training_history_per_class.csv")

print("=== CHECKPOINT VALIDATION ===")
ckpt = torch.load(CKPT, map_location="cpu")
print("  Keys             :", list(ckpt.keys()))
print("  epoch            :", ckpt.get("epoch"))
print("  best_macro_f1    :", round(ckpt.get("best_macro_f1", 0), 4))
print("  architecture     :", ckpt.get("architecture"))
norm = ckpt.get("normalization", {})
print("  normalization mean:", norm.get("mean"))
print("  normalization std :", norm.get("std"))
cw = ckpt.get("class_weights", [])
print("  class_weights[8] (potato):", cw[8] if len(cw) > 8 else "N/A")
print("  optimizer_state  : present =", ("optimizer_state" in ckpt))
print("  scaler_state     : present =", ("scaler_state" in ckpt))

assert ckpt.get("epoch") == 2, "Expected epoch=2"
assert abs(ckpt.get("best_macro_f1", 0) - 0.7415) < 0.002, "Unexpected best_f1"
print("  Guard-rail PASSED: epoch==2, best_f1~=0.7415")

print()
print("=== HISTORY FILE VALIDATION ===")
df = pd.read_csv(HIST)
print("  Rows in history  :", len(df))
print("  Epochs present   :", df["epoch"].tolist())
assert list(df["epoch"]) == [1, 2], "Unexpected epochs"
print("  History guard-rail PASSED")

df_pc = pd.read_csv(PC)
ep2 = df_pc[df_pc["epoch"] == 2].iloc[0]
print("  Potato Ep2: Prec={:.4f} Rec={:.4f} F1={:.4f}".format(
    ep2["potato_precision"], ep2["potato_recall"], ep2["potato_f1"]))
print("  Tomato Ep2: Prec={:.4f} Rec={:.4f} F1={:.4f}".format(
    ep2["tomato_precision"], ep2["tomato_recall"], ep2["tomato_f1"]))

print()
print("=== ARCHITECTURE VALIDATION ===")
from src.models.model2_14class_resnet9 import create_model2_14class_resnet9
model = create_model2_14class_resnet9()
model.load_state_dict(ckpt["model_state"])
print("  model.load_state_dict : OK")
total_params = sum(p.numel() for p in model.parameters())
print("  Total parameters      :", f"{total_params:,}")
last_children = list(model.children())
print("  Last module           :", last_children[-1])

print()
print("=== DEVICE CHECK ===")
print("  CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU            :", torch.cuda.get_device_name(0))

print()
print("=== SCRIPT VALIDATION ===")
script = os.path.join(BASE_DIR, "resume_model2_14class_5_more_epochs.py")
print("  Script exists  :", os.path.exists(script))
content = open(script).read()
print("  Script length  :", len(content), "chars")
checks = {
    "FINAL_EPOCH = 7":       "FINAL_EPOCH        = 7",
    "NEXT_EPOCH = 3":        "NEXT_EPOCH         = 3",
    "RESUME_LR = 5e-4":      "RESUME_LR          = 5e-4",
    "ReduceLROnPlateau":      "ReduceLROnPlateau",
    "append only (mode a)":  'HISTORY_FILE, "a"',
    "no history clear":      'os.remove(self.history_file)' not in content,
    "best F1 guard":         "best_f1    = macro_f1",
    "confusion matrix":      "generate_confusion_matrix",
    "report writer":         "write_report",
}
all_ok = True
for label, check in checks.items():
    if isinstance(check, bool):
        ok = check
    else:
        ok = check in content
    status = "OK" if ok else "MISSING"
    if not ok:
        all_ok = False
    print("  {:35s}: {}".format(label, status))

print()
if all_ok:
    print("ALL VALIDATIONS PASSED. Script is ready for manual execution.")
else:
    print("SOME CHECKS FAILED - review above.")
