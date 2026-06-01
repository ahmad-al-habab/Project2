import os
import json
import cv2
import random
from tqdm import tqdm

IMAGE_DIR = "raw_data/images"
LABEL_DIR = "raw_data/labels_labelme"

OUT_IMAGE_DIR = "dataset_tiled_yolo/images"
OUT_LABEL_DIR = "dataset_tiled_yolo/labels"

TILE_SIZE = 640
OVERLAP = 160
BACKGROUND_KEEP_PROB = 0.20

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

random.seed(42)


def normalize_polygon(points, x0, y0, tile_size):
    normalized = []
    for x, y in points:
        nx = (x - x0) / tile_size
        ny = (y - y0) / tile_size
        normalized.extend([nx, ny])
    return normalized


def polygon_inside_tile(points, x0, y0, tile_size):
    for x, y in points:
        if x < x0 or x > x0 + tile_size or y < y0 or y > y0 + tile_size:
            return False
    return True


json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".json")]

total_tiles = 0
positive_tiles = 0
background_tiles = 0

for json_file in tqdm(json_files):
    json_path = os.path.join(LABEL_DIR, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data["imagePath"]
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_name}")
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"Cannot read image: {image_name}")
        continue

    h, w = image.shape[:2]

    shapes = [
        shape for shape in data["shapes"]
        if shape["label"] == "tank" and shape["shape_type"] == "polygon"
    ]

    step = TILE_SIZE - OVERLAP

    base_name = os.path.splitext(image_name)[0]

    for y0 in range(0, h - TILE_SIZE + 1, step):
        for x0 in range(0, w - TILE_SIZE + 1, step):

            tile = image[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]

            yolo_lines = []

            for shape in shapes:
                points = shape["points"]

                if polygon_inside_tile(points, x0, y0, TILE_SIZE):
                    norm_points = normalize_polygon(points, x0, y0, TILE_SIZE)

                    if all(0 <= p <= 1 for p in norm_points):
                        line = "0 " + " ".join([f"{p:.6f}" for p in norm_points])
                        yolo_lines.append(line)

            is_positive = len(yolo_lines) > 0

            if not is_positive:
                if random.random() > BACKGROUND_KEEP_PROB:
                    continue

            tile_name = f"{base_name}_{x0}_{y0}.jpg"
            label_name = f"{base_name}_{x0}_{y0}.txt"

            cv2.imwrite(os.path.join(OUT_IMAGE_DIR, tile_name), tile)

            with open(os.path.join(OUT_LABEL_DIR, label_name), "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")

            total_tiles += 1
            if is_positive:
                positive_tiles += 1
            else:
                background_tiles += 1

print("Tiling completed!")
print(f"Total tiles: {total_tiles}")
print(f"Positive tiles: {positive_tiles}")
print(f"Background tiles: {background_tiles}")