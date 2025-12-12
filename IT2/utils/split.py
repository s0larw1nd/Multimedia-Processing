import os
import shutil
import random

images_dir="dataset/images"
labels_dir="dataset/labels"
output_dir="data"
train_percent=90

train_img_dir = os.path.join(output_dir, "train/images")
train_lbl_dir = os.path.join(output_dir, "train/labels")
val_img_dir = os.path.join(output_dir, "validation/images")
val_lbl_dir = os.path.join(output_dir, "validation/labels")

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(images_dir) if f.endswith(".png")]
images.sort()

random.shuffle(images)

train_count = int(len(images) * train_percent / 100)

train_files = images[:train_count]
val_files = images[train_count:]

def copy_pair(filename, target_img_dir, target_lbl_dir):
    img_src = os.path.join(images_dir, filename)
    lbl_src = os.path.join(labels_dir, filename.replace(".png", ".txt"))

    img_dst = os.path.join(target_img_dir, filename)
    lbl_dst = os.path.join(target_lbl_dir, filename.replace(".png", ".txt"))

    shutil.copy(img_src, img_dst)

    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, lbl_dst)
    else:
        print(f"Нет метки для {filename}: {lbl_src}")

for f in train_files:
    copy_pair(f, train_img_dir, train_lbl_dir)

for f in val_files:
    copy_pair(f, val_img_dir, val_lbl_dir)