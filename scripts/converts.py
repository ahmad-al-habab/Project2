import os
import json
from tqdm import tqdm

# المسارات
IMAGE_DIR = "raw_data/images"
LABEL_DIR = "raw_data/labels_labelme"

OUTPUT_LABEL_DIR = "dataset_yolo/labels"
OUTPUT_IMAGE_DIR = "dataset_yolo/images"

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

def normalize(points, img_w, img_h):
    norm_points = []
    for x, y in points:
        norm_points.append(x / img_w)
        norm_points.append(y / img_h)
    return norm_points

for json_file in tqdm(os.listdir(LABEL_DIR)):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(LABEL_DIR, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data["imagePath"]
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    txt_filename = os.path.splitext(json_file)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_filename)

    with open(txt_path, "w") as out_file:
        for shape in data["shapes"]:
            if shape["label"] != "tank":
                continue

            points = shape["points"]
            norm_points = normalize(points, img_w, img_h)

            line = "0 " + " ".join(map(str, norm_points))
            out_file.write(line + "\n")

    # نسخ الصورة
    src_img = os.path.join(IMAGE_DIR, image_name)
    dst_img = os.path.join(OUTPUT_IMAGE_DIR, image_name)

    if os.path.exists(src_img):
        import shutil
        shutil.copy(src_img, dst_img)

print("Conversion Done!")