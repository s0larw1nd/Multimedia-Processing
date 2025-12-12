import os
from PIL import Image

input_dir = "./frames_old"
output_dir = "./frames"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        with Image.open(in_path) as img:
            width, height = img.size
            cut = height // 7

            cropped = img.crop((0, cut, width, height - cut))
            cropped.save(out_path)