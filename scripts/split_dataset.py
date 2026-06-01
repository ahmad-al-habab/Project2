import os
import shutil
from sklearn.model_selection import train_test_split

IMAGE_DIR = "dataset_tiled_yolo/images"
LABEL_DIR = "dataset_tiled_yolo/labels"

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png")]

train, temp = train_test_split(images, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

def move_files(file_list, split):
    os.makedirs(f"{IMAGE_DIR}/{split}", exist_ok=True)
    os.makedirs(f"{LABEL_DIR}/{split}", exist_ok=True)

    for file in file_list:
        img_src = os.path.join(IMAGE_DIR, file)
        lbl_src = os.path.join(LABEL_DIR, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img_dst = os.path.join(IMAGE_DIR, split, file)
        lbl_dst = os.path.join(LABEL_DIR, split, os.path.basename(lbl_src))

        shutil.move(img_src, img_dst)
        shutil.move(lbl_src, lbl_dst)

move_files(train, "train")
move_files(val, "val")
move_files(test, "test")

print("Dataset split completed!")