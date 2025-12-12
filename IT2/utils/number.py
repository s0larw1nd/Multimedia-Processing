import os
import re

directory = "./frames"
pattern = re.compile(r"frame_(\d+)\.png$")

files = []


for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        num = int(match.group(1))
        files.append((num, filename))

files.sort(key=lambda x: x[0])

for i, (_, filename) in enumerate(files):
    old_path = os.path.join(directory, filename)
    new_name = f"frame_{i}.png"
    temp_path = os.path.join(directory, f"temp_{i}.png")
    new_path = os.path.join(directory, new_name)

    os.rename(old_path, temp_path)

for i, _ in enumerate(files):
    temp_path = os.path.join(directory, f"temp_{i}.png")
    final_path = os.path.join(directory, f"frame_{i}.png")
    os.rename(temp_path, final_path)
