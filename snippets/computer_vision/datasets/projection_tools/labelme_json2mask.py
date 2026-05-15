# labelme  .json  ->  palette<mask> .png   单通道、调色板彩色图
from PIL import Image, ImageDraw
from collections import defaultdict
import os, json, base64

# 定义类别和对应的颜色
classes = ["_background_", "jyz", "baodian"]
colors = {
    "_background_": (0, 0, 0),
    "jyz": (255, 0, 0),      # 红色
    "baodian": (0, 255, 0)   # 绿色
}

def json2mask(json_path, mask_path, classes, colors):
    with open(json_path, 'r', encoding='utf-8')as f:
        data = json.load(f)
    
    img_width = data.get('imageWidth', 0)
    img_height = data.get('imageHeight', 0)
    if img_width == 0 or img_height ==0:
        raise ValueError(f"Invalid image size in {json_path}")

    mask = Image.new('P', (img_width, img_height), color=0)
    
    palette = []
    for cls in classes:
        palette.extend(colors[cls])
    palette += [0,0,0] * (256-len(classes))
    mask.putpalette(palette)
    
    draw = ImageDraw.Draw(mask)
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in classes:
            print(f"Unknown label '{label}' in {json_path}, skipping...")
            continue
        points = shape.get('points', [])
        if not points:
            continue
        polygon = [tuple(map(int, point)) for point in points]
        class_index = classes.index(label)
        draw.polygon(polygon, fill=class_index)
    
    mask.save(mask_path, format='PNG')
  
def convert_dataset(data_root, output_dir)  :
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(data_root):
        if filename.endswith('.json'):
            json_path = os.path.join(data_root, filename)
            mask_filename = filename.replace('.json', '.png')
            mask_path = os.path.join(output_dir, mask_filename)
            json2mask(json_path, mask_path)
       
# get jpg from json 
def get_jpg_json(jsonpath, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_filename = os.path.basename(jsonpath)
    jpg_filename = os.path.splitext(json_filename)[0] +'.jpg'
    jpg_path = os.path.join(output_dir, jpg_filename)
    
    with open(jsonpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_data = data.get('imageData')
    image_bytes = base64.b64decode(image_data)
    with open(jpg_path, 'wb') as img_f:
        img_f.write(image_bytes)
    





