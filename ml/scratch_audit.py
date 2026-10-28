import json
import os

in_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\scratch_class_counts.json"
out_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports"
out_path = os.path.join(out_dir, "caltech256_category_audit.txt")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(in_path, "r") as f:
    class_counts = json.load(f)

# Explicit rules for exclusions
exclude_dict = {
    "015.bonsai-101": "Potted plant/tree",
    "025.cactus": "Plant",
    "068.fern": "Plant/foliage",
    "092.grapes": "Plant part/fruit, often photographed with leaves",
    "103.hibiscus": "Flower/botanical",
    "118.iris": "Flower/botanical",
    "147.mushroom": "Fungi/botanical-adjacent, often in grass",
    "154.palm-tree": "Tree/plant",
    "204.sunflower-101": "Flower/botanical",
    "221.tomato": "Plant part/fruit, often photographed with leaves",
    "242.watermelon": "Plant part/fruit",
    "257.clutter": "Ambiguous meaning (random background clutter)",
    "024.butterfly": "Insect often photographed on flowers/leaves",
    "093.grasshopper": "Insect often photographed on leaves/grass",
    "113.hummingbird": "Bird often photographed interacting with flowers",
    "166.praying-mantis": "Insect often camouflaged on leaves",
    "189.snail": "Often photographed on leaves"
}

safe_list = []
safe_count = 0
exclude_list = []
exclude_count = 0

for cls, count in sorted(class_counts.items()):
    if cls in exclude_dict:
        exclude_list.append((cls, count, exclude_dict[cls]))
        exclude_count += count
    else:
        safe_list.append((cls, count))
        safe_count += count

with open(out_path, "w") as f:
    f.write(f"Total categories: {len(class_counts)}\n\n")
    
    f.write("SAFE_NON_LEAF category list:\n")
    for cls, count in safe_list:
        f.write(f"- {cls} ({count} images)\n")
    
    f.write(f"\nSAFE_NON_LEAF total image count: {safe_count}\n\n")
    
    f.write("EXCLUDE_OR_REVIEW category list:\n")
    for cls, count, reason in exclude_list:
        f.write(f"- {cls} ({count} images) - Reason: {reason}\n")
    
    f.write(f"\nEXCLUDE_OR_REVIEW total image count: {exclude_count}\n")

print(f"Audit complete. Report generated at {out_path}")
print(f"Total categories: {len(class_counts)}")
print(f"SAFE_NON_LEAF categories: {len(safe_list)}")
print(f"SAFE_NON_LEAF total images: {safe_count}")
print(f"EXCLUDE_OR_REVIEW categories: {len(exclude_list)}")
print(f"EXCLUDE_OR_REVIEW total images: {exclude_count}")
