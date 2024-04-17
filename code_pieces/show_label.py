import os
from tqdm import tqdm
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

def draw_mask_on_jpg(png_folder, jpg_folder, save_folder, resize=None, mode='opencv'):
    """
    Draw PNG mask boundaries on corresponding JPG images.

    Args:
        png_folder (str): Directory path containing PNG mask files.
        jpg_folder (str): Directory path containing JPG files.
        save_folder (str): Directory path to save annotated images.
        resize (tuple, optional): Target dimensions (width, height) to resize images (if desired).
        mode (str): Library to use for image processing ('opencv' or 'pillow').
    """
    for mask_name in tqdm(os.listdir(png_folder)):
        mask_path = os.path.join(png_folder, mask_name)
        jpg_path = os.path.join(jpg_folder, mask_name.replace('.png', '.jpg'))
        assert os.path.exists(mask_path), f"{mask_path} does not exist"
        assert os.path.exists(jpg_path), f"{jpg_path} does not exist"

        if mode == 'opencv':
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(jpg_path)
            if resize:
                image = cv2.resize(image, resize)
                mask = cv2.resize(mask, resize)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        elif mode == 'pillow':
            mask = Image.open(mask_path).convert("L")
            image = Image.open(jpg_path)
            if resize:
                image = image.resize(resize)
                mask = mask.resize(resize)
            draw = ImageDraw.Draw(image)
            contours, _ = cv2.findContours(np.array(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                poly = [tuple(point[0]) for point in contour]
                draw.line(poly + [poly[0]], fill=(0, 255, 0), width=2)

        save_path = os.path.join(save_folder, mask_name.replace('.png', '.jpg'))
        if mode == 'opencv':
            cv2.imwrite(save_path, image)
        elif mode == 'pillow':
            image.save(save_path, "JPEG")
    """
    Args:
        png_folder (str:dir_path): jpg_folder
        jpg_folder (str:dir_path): png_folder
        save_folder (str:dir_path): draw png(binary) boundaries on jpg    -->   save_folder
        resize (optional: None or tuple(target_width, target_height)): whether resize jpg png
    """
    
    for mask_name in tqdm(os.listdir(png_folder)):
        mask_path = os.path.join(png_folder, mask_name)
        jpg_path = os.path.join(jpg_folder, mask_name.replace('.png', '.jpg'))
        assert os.path.exists(mask_path), f"{mask_path} not exists"
        assert os.path.exists(jpg_path), f"{jpg_path} not exists"

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(jpg_path)

        if resize is not None:
            assert resize is None or (isinstance(resize, tuple) and len(resize) == 2), f"{resize} is in a wrong format"
            image = cv2.resize(image, resize)
            mask = cv2.resize(mask, resize)

        countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, countours, -1, (0, 255, 0), 2)

        save_path = os.path.join(save_folder, mask_name.replace('.png', '.jpg'))
        cv2.imwrite(save_path, image)

def draw_json_on_jpg(json_folder, jpg_folder, save_folder, resize=None, mode='opencv'):
    """
    Draw annotations defined in JSON files on corresponding JPG images.

    Args:
        json_folder (str): Directory path containing JSON files with annotations.
        jpg_folder (str): Directory path containing JPG files.
        save_folder (str): Directory path to save annotated images.
        resize (tuple, optional): Target dimensions (width, height) to resize images (if desired).
        mode (str): Library to use for image processing ('opencv' or 'pillow').
    """
    for json_name in tqdm(os.listdir(json_folder)):
        json_path = os.path.join(json_folder, json_name)
        jpg_path = os.path.join(jpg_folder, json_name.replace('.json', '.jpg'))
        assert os.path.exists(json_path), f"{json_path} does not exist"
        assert os.path.exists(jpg_path), f"{jpg_path} does not exist"

        if mode == 'opencv':
            image = cv2.imread(jpg_path)
            if resize:
                image = cv2.resize(image, resize)
        elif mode == 'pillow':
            image = Image.open(jpg_path)
            if resize:
                image = image.resize(resize)
            draw = ImageDraw.Draw(image)

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        for annotation in annotations:
            points = annotation['points']
            if annotation['shape_type'] == 'polygon':
                if mode == 'opencv':
                    contours = np.array(points, dtype=np.int32)
                    cv2.polylines(image, [contours], isClosed=True, color=(0, 255, 0), thickness=2)
                elif mode == 'pillow':
                    draw.polygon(points, outline=(0, 255, 0))
            elif annotation['shape_type'] == 'rectangle':
                pt1, pt2 = tuple(points[0]), tuple(points[1])
                if mode == 'opencv':
                    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
                elif mode == 'pillow':
                    draw.rectangle([pt1, pt2], outline=(0, 255, 0))

        save_path = os.path.join(save_folder, json_name.replace('.json', '.jpg'))
        if mode == 'opencv':
            cv2.imwrite(save_path, image)
        elif mode == 'pillow':
            image.save(save_path, "JPEG")






