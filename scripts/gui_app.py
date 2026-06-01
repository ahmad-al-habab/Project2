import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "runs/segment/train-8/weights/best.pt"

CONF = 0.45
IOU = 0.5

model = YOLO(MODEL_PATH)


def process_image(file_path):
    try:
        # تحميل الصورة وتصغيرها
        img = cv2.imread(file_path)
        img = cv2.resize(img, (1024, 1024))  # 🔥 مهم

        results = model(
            source=img,
            conf=CONF,
            iou=IOU,
            retina_masks=True
        )

        result = results[0]

        # عدد التانكات
        tank_count = len(result.boxes) if result.boxes is not None else 0

        # متوسط الثقة
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            avg_conf = np.mean(confs)
        else:
            avg_conf = 0.0

        # صورة بعد المعالجة
        plotted = result.plot()
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        return plotted, tank_count, avg_conf

    except Exception as e:
        print("Error:", e)
        return None, 0, 0


def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    # الصورة الأصلية
    original = Image.open(file_path).convert("RGB")
    original.thumbnail((500, 500))
    original_tk = ImageTk.PhotoImage(original)

    original_label.config(image=original_tk)
    original_label.image = original_tk

    # معالجة الصورة
    processed, count, avg_conf = process_image(file_path)

    if processed is not None:
        img = Image.fromarray(processed)
        img.thumbnail((500, 500))
        processed_tk = ImageTk.PhotoImage(img)

        result_label.config(image=processed_tk)
        result_label.image = processed_tk

    count_label.config(text=f"Tanks: {count}")
    conf_label.config(text=f"Avg Confidence: {avg_conf:.2f}")


# ===== GUI =====

root = tk.Tk()
root.title("Oil Tank Detection - YOLOv8")
root.geometry("1200x700")
root.configure(bg="#121212")

title = tk.Label(
    root,
    text="Oil Storage Tank Detection",
    font=("Arial", 24, "bold"),
    fg="white",
    bg="#121212"
)
title.pack(pady=10)

btn = tk.Button(
    root,
    text="Select Image",
    command=select_image,
    font=("Arial", 14),
    bg="#0078D7",
    fg="white",
    padx=15,
    pady=5
)
btn.pack(pady=10)

info_frame = tk.Frame(root, bg="#121212")
info_frame.pack()

count_label = tk.Label(
    info_frame,
    text="Tanks: 0",
    font=("Arial", 14),
    fg="white",
    bg="#121212"
)
count_label.pack(side="left", padx=20)

conf_label = tk.Label(
    info_frame,
    text="Avg Confidence: 0.00",
    font=("Arial", 14),
    fg="white",
    bg="#121212"
)
conf_label.pack(side="left", padx=20)

# إطار الصور
image_frame = tk.Frame(root, bg="#121212")
image_frame.pack(pady=20)

original_label = tk.Label(image_frame, bg="#121212")
original_label.pack(side="left", padx=20)

result_label = tk.Label(image_frame, bg="#121212")
result_label.pack(side="right", padx=20)

root.mainloop()