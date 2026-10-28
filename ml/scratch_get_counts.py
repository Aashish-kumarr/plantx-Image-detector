import os
import json

caltech_root = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\caltech256\256_ObjectCategories"
class_counts = {}

for entry in os.scandir(caltech_root):
    if entry.is_dir():
        count = 0
        for file in os.scandir(entry.path):
            if file.is_file() and file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
        class_counts[entry.name] = count

out_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\scratch_class_counts.json"
with open(out_path, "w") as f:
    json.dump(class_counts, f)

print(f"Total categories: {len(class_counts)}")
