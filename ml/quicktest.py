import os, sys, pandas as pd
from PIL import Image

BASE_DIR = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
sys.path.insert(0, BASE_DIR)

OLD_PATH_PREFIX = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
NEW_PATH_PREFIX = BASE_DIR

def remap_image_path(raw_path):
    if raw_path.lower().startswith(OLD_PATH_PREFIX.lower()):
        return NEW_PATH_PREFIX + raw_path[len(OLD_PATH_PREFIX):]
    return raw_path

CSV_PATH = os.path.join(BASE_DIR, "metadata", "model2_14class_split.csv")
df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)

# Test first 5 image paths
print("Path remap smoke test:")
for i in range(5):
    raw  = str(train_df.iloc[i]["image_path"])
    new  = remap_image_path(raw)
    ok   = os.path.exists(new)
    print(f"  [{i+1}] exists={ok}  {os.path.basename(new)}")
    if not ok:
        print(f"       MISSING: {new}")

# Try opening first image
row = train_df.iloc[0]
img_path = remap_image_path(str(row["image_path"]))
pil = Image.open(img_path).convert("RGB")
print(f"\nFirst image opened OK: {pil.size}")
print("PATH REMAP WORKING")
