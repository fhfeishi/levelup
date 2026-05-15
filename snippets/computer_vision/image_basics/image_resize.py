import os
from PIL import Image
from tqdm import tqdm

jpg_dir = r"F:\bianse\1280_960_dataset\conservator"
names = [name for name in os.listdir(jpg_dir) if name.endswith(".jpg")]
for name in tqdm(names):
    path = f"{jpg_dir}/{name}"
    image = Image.open(path).convert("RGB")
    image_new = image.resize((1280,960))
    image_new.save(path)
