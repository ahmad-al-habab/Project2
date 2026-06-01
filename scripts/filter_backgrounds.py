import os
import random

SPLITS = ["train", "val"]
BACKGROUND_RATIO = 0.30  # نخلي الخلفيات حوالي 30%

random.seed(42)
for split in SPLITS:
    label_dir = f"dataset_yolo/labels/{split}"
    image_dir = f"dataset_yolo/images/{split}"

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    positive = []
    background = []

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        if os.path.getsize(label_path) == 0:
            background.append(label_file)
        else:
            positive.append(label_file)

    max_background = int(len(positive) * BACKGROUND_RATIO)

    backgrounds_to_keep = set(random.sample(background, min(len(background), max_background)))
    backgrounds_to_remove = [f for f in background if f not in backgrounds_to_keep]

    for label_file in backgrounds_to_remove:
        label_path = os.path.join(label_dir, label_file)

        image_name_jpg = label_file.replace(".txt", ".jpg")
        image_name_png = label_file.replace(".txt", ".png")

        image_path_jpg = os.path.join(image_dir, image_name_jpg)
        image_path_png = os.path.join(image_dir, image_name_png)

        os.remove(label_path)

        if os.path.exists(image_path_jpg):
            os.remove(image_path_jpg)

        if os.path.exists(image_path_png):
            os.remove(image_path_png)

    print(f"{split}:")
    print(f"  positive kept: {len(positive)}")
    print(f"  background kept: {len(backgrounds_to_keep)}")
    print(f"  background removed: {len(backgrounds_to_remove)}")