import os
import shutil
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
src_folder = r"dataset/Stroke_classification"  # original dataset folder
dst_folder = r"dataset/data"                 # train/test folder destination

# -----------------------------
# Classes mapping
# -----------------------------
classes = {
    "non-stroke": "normal",
    "stroke": ["haemorrhagic", "ischemic"]
}

# Create train/test folders
os.makedirs(os.path.join(dst_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(dst_folder, "test"), exist_ok=True)

for label, folders in classes.items():
    os.makedirs(os.path.join(dst_folder, "train", label), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "test", label), exist_ok=True)

    # Collect all image paths
    if isinstance(folders, list):
        all_images = []
        for f in folders:
            folder_path = os.path.join(src_folder, f)
            all_images += [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    else:
        folder_path = os.path.join(src_folder, folders)
        all_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

    # Split 80% train / 20% test
    train_imgs, test_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

    # Copy images to train/test folders
    for img_path in train_imgs:
        shutil.copy(img_path, os.path.join(dst_folder, "train", label))
    for img_path in test_imgs:
        shutil.copy(img_path, os.path.join(dst_folder, "test", label))

print("✅ Dataset split into train/test folders successfully!")
