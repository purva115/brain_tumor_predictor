# scripts/check_data.py
import os

root = "data/BrainMRI"
if not os.path.exists(root):
    print("ERROR: data/BrainMRI does not exist. Please download/unzip dataset as instructed.")
    exit(1)

for label in ["no", "yes"]:
    folder = os.path.join(root, label)
    if not os.path.exists(folder):
        print(f"WARNING: folder missing: {folder}")
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"{label}: {len(files)} images  (example: {files[:3]})")

total = sum(len([f for f in os.listdir(os.path.join(root, l)) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            for l in ["no","yes"] if os.path.exists(os.path.join(root,l)))
print(f"Total images found: {total}")
