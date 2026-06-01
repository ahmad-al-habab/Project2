import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from torchvision.ops import nms
import os
import time
from datetime import datetime

MODEL_PATH = "runs/segment/train-8/weights/best.pt"

CONF = 0.50
IOU = 0.50
TILE_SIZE = 640
OVERLAP = 160

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

current_image_path = None
current_original = None
current_result = None

BG = "#0B1220"
SIDEBAR = "#111827"
PANEL = "#162033"
BORDER = "#263449"
BLUE = "#3B82F6"
RED = "#EF4444"
GREEN = "#22C55E"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
YELLOW = "#FACC15"


class ZoomCanvas:
    def __init__(self, parent, placeholder):
        self.frame = tk.Frame(parent, bg="#0F172A")
        self.frame.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        self.canvas = tk.Canvas(
            self.frame,
            bg="#0F172A",
            highlightthickness=0,
            cursor="fleur"
        )
        self.canvas.pack(fill="both", expand=True)

        self.image_rgb = None
        self.tk_img = None
        self.scale = 1.0
        self.placeholder = placeholder

        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        self.canvas.bind("<Button-4>", self.mouse_zoom)
        self.canvas.bind("<Button-5>", self.mouse_zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

        self.show_placeholder()

    def show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            300,
            220,
            text=self.placeholder,
            fill=MUTED,
            font=("Arial", 18, "bold")
        )

    def set_image(self, image_rgb):
        self.image_rgb = image_rgb
        self.scale = 1.0
        self.update_image()

    def clear(self):
        self.image_rgb = None
        self.tk_img = None
        self.scale = 1.0
        self.show_placeholder()

    def update_image(self):
        self.canvas.delete("all")

        if self.image_rgb is None:
            self.show_placeholder()
            return

        img = Image.fromarray(self.image_rgb)

        cw = max(self.canvas.winfo_width(), 600)
        ch = max(self.canvas.winfo_height(), 430)

        iw, ih = img.size
        base_scale = min(cw / iw, ch / ih) * 0.96
        final_scale = base_scale * self.scale

        new_w = int(iw * final_scale)
        new_h = int(ih * final_scale)

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        x = max((cw - new_w) // 2, 0)
        y = max((ch - new_h) // 2, 0)

        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def mouse_zoom(self, event):
        if self.image_rgb is None:
            return

        if event.delta > 0 or event.num == 4:
            self.scale *= 1.15
        else:
            self.scale /= 1.15

        self.scale = max(0.25, min(self.scale, 6.0))
        self.update_image()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


def run_tiled_inference(image_bgr):
    h, w = image_bgr.shape[:2]
    result_image = image_bgr.copy()

    all_boxes = []
    all_scores = []

    step = TILE_SIZE - OVERLAP

    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)

            tile = image_bgr[y:y2, x:x2]

            if tile.shape[0] < 200 or tile.shape[1] < 200:
                continue

            padded_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            padded_tile[:tile.shape[0], :tile.shape[1]] = tile

            results = model(
                source=padded_tile,
                conf=CONF,
                iou=IOU,
                retina_masks=True,
                device=device,
                verbose=False
            )

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2b, y2b = box.astype(int)

                if x1 >= tile.shape[1] or y1 >= tile.shape[0]:
                    continue

                x1 = max(0, min(x1, tile.shape[1] - 1))
                y1 = max(0, min(y1, tile.shape[0] - 1))
                x2b = max(0, min(x2b, tile.shape[1] - 1))
                y2b = max(0, min(y2b, tile.shape[0] - 1))

                all_boxes.append([x + x1, y + y1, x + x2b, y + y2b])
                all_scores.append(float(confs[i]))

    if len(all_boxes) == 0:
        return result_image, 0, 0.0

    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.25)

    final_boxes = boxes_tensor[keep].numpy()
    final_scores = scores_tensor[keep].numpy()

    filtered_boxes = []
    filtered_scores = []

    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        ratio = box_w / (box_h + 1e-6)

        if 0.65 < ratio < 1.45 and 18 < box_w < 320 and 18 < box_h < 320:
            filtered_boxes.append(box)
            filtered_scores.append(final_scores[i])

    final_boxes = np.array(filtered_boxes)
    final_scores = np.array(filtered_scores)

    if len(final_boxes) == 0:
        return result_image, 0, 0.0

    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = final_scores[i]

        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"Oil Tank {conf:.2f}"
        cv2.rectangle(result_image, (x1, y1 - 28), (x1 + 145, y1), (0, 0, 255), -1)

        cv2.putText(
            result_image,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    total_count = len(final_boxes)
    avg_conf = float(np.mean(final_scores)) if len(final_scores) > 0 else 0.0

    return result_image, total_count, avg_conf


def open_image():
    global current_image_path, current_original, current_result

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    image_bgr = cv2.imread(file_path)

    if image_bgr is None:
        messagebox.showerror("Error", "Cannot read image")
        return

    current_image_path = file_path
    current_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    current_result = None

    original_viewer.set_image(current_original)
    result_viewer.clear()

    file_name_value.config(text=os.path.basename(file_path))
    resolution_value.config(text=f"{image_bgr.shape[1]} x {image_bgr.shape[0]}")
    date_value.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    detections_top_value.config(text="0")
    total_value.config(text="0")
    avg_value.config(text="0.00")
    time_value.config(text="0.00 s")
    status_value.config(text="● Image Loaded", fg=YELLOW)


def run_detection():
    global current_result

    if current_image_path is None:
        messagebox.showwarning("Warning", "Please select an image first.")
        return

    image_bgr = cv2.imread(current_image_path)

    if image_bgr is None:
        messagebox.showerror("Error", "Cannot read image")
        return

    status_value.config(text="● Processing...", fg=YELLOW)
    root.update()

    start = time.time()
    processed_bgr, count, avg_conf = run_tiled_inference(image_bgr)
    processing_time = time.time() - start

    current_result = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    result_viewer.set_image(current_result)

    detections_top_value.config(text=str(count))
    total_value.config(text=str(count))
    avg_value.config(text=f"{avg_conf:.2f}")
    time_value.config(text=f"{processing_time:.2f} s")

    status_value.config(text="● Ready", fg=GREEN)


def clear_all():
    global current_image_path, current_original, current_result

    current_image_path = None
    current_original = None
    current_result = None

    original_viewer.clear()
    result_viewer.clear()

    file_name_value.config(text="No image selected")
    resolution_value.config(text="-")
    date_value.config(text="-")
    detections_top_value.config(text="0")
    total_value.config(text="0")
    avg_value.config(text="0.00")
    time_value.config(text="0.00 s")
    status_value.config(text="● Ready", fg=GREEN)


def button(parent, text, command, bg=BLUE):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Arial", 12, "bold"),
        bg=bg,
        fg="white",
        activebackground=bg,
        activeforeground="white",
        relief="flat",
        padx=15,
        pady=12,
        cursor="hand2"
    )


def make_panel(parent):
    return tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)


def stat_box(parent, icon, title, value_color):
    frame = tk.Frame(parent, bg=PANEL)
    frame.pack(side="left", fill="both", expand=True, padx=8)

    tk.Label(frame, text=icon, font=("Arial", 25, "bold"), fg=value_color, bg=PANEL).pack(side="left", padx=15)

    inner = tk.Frame(frame, bg=PANEL)
    inner.pack(side="left", pady=12)

    tk.Label(inner, text=title, font=("Arial", 10, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w")

    val = tk.Label(inner, text="-", font=("Arial", 18, "bold"), fg=value_color, bg=PANEL)
    val.pack(anchor="w")

    return val


root = tk.Tk()
root.title("Detection Viewer")
root.geometry("1500x920")
root.configure(bg=BG)

topbar = tk.Frame(root, bg="#07111F", height=90)
topbar.pack(fill="x")
topbar.pack_propagate(False)

tk.Label(
    topbar,
    text="◎",
    font=("Arial", 38, "bold"),
    fg=TEXT,
    bg="#07111F"
).pack(side="left", padx=(25, 10))

title_box = tk.Frame(topbar, bg="#07111F")
title_box.pack(side="left")

tk.Label(
    title_box,
    text="Detection Viewer",
    font=("Arial", 22, "bold"),
    fg=TEXT,
    bg="#07111F"
).pack(anchor="w", pady=(18, 0))

tk.Label(
    title_box,
    text="Object Detection System",
    font=("Arial", 11),
    fg=MUTED,
    bg="#07111F"
).pack(anchor="w")

body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True)

sidebar = make_panel(body)
sidebar.config(width=285)
sidebar.pack(side="left", fill="y", padx=(18, 10), pady=22)
sidebar.pack_propagate(False)

tk.Label(sidebar, text="CONTROLS", font=("Arial", 12, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22, pady=(25, 12))

button(sidebar, "📁  Open Image", open_image, BLUE).pack(fill="x", padx=22, pady=7)
button(sidebar, "▶  Run Detection", run_detection, BLUE).pack(fill="x", padx=22, pady=7)
button(sidebar, "🗑  Clear", clear_all, "#1E293B").pack(fill="x", padx=22, pady=7)

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=18, pady=25)

tk.Label(sidebar, text="MODEL INFO", font=("Arial", 12, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22)

tk.Label(sidebar, text="Model Name:", font=("Arial", 11), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22, pady=(20, 2))
tk.Label(sidebar, text=os.path.basename(MODEL_PATH), font=("Arial", 11, "bold"), fg=BLUE, bg=PANEL).pack(anchor="w", padx=22)

tk.Label(sidebar, text="Classes:", font=("Arial", 11), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22, pady=(25, 2))
tk.Label(sidebar, text="●  Oil Tank", font=("Arial", 12, "bold"), fg=RED, bg=PANEL).pack(anchor="w", padx=22)

tk.Label(sidebar, text="Mouse Controls:", font=("Arial", 11, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22, pady=(25, 2))
tk.Label(
    sidebar,
    text="Wheel = Zoom In / Out\nLeft Click + Drag = Move",
    font=("Arial", 10),
    fg=MUTED,
    bg=PANEL,
    justify="left"
).pack(anchor="w", padx=22)

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=18, pady=25)

tk.Label(sidebar, text="STATUS", font=("Arial", 12, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w", padx=22)

status_value = tk.Label(sidebar, text="● Ready", font=("Arial", 13, "bold"), fg=GREEN, bg=PANEL)
status_value.pack(anchor="w", padx=22, pady=12)

main = tk.Frame(body, bg=BG)
main.pack(side="left", fill="both", expand=True, padx=(5, 18), pady=22)

info_bar = make_panel(main)
info_bar.pack(fill="x", pady=(0, 15))


def info_item(parent, icon, title, value, color=BLUE):
    item = tk.Frame(parent, bg=PANEL)
    item.pack(side="left", fill="both", expand=True, padx=18, pady=12)

    tk.Label(item, text=icon, font=("Arial", 24, "bold"), fg=MUTED, bg=PANEL).pack(side="left", padx=(0, 12))

    texts = tk.Frame(item, bg=PANEL)
    texts.pack(side="left")

    tk.Label(texts, text=title, font=("Arial", 10, "bold"), fg=TEXT, bg=PANEL).pack(anchor="w")
    val = tk.Label(texts, text=value, font=("Arial", 12, "bold"), fg=color, bg=PANEL)
    val.pack(anchor="w")

    return val


file_name_value = info_item(info_bar, "▧", "Image Name", "No image selected")
resolution_value = info_item(info_bar, "↙", "Resolution", "-")
date_value = info_item(info_bar, "▣", "Date", "-")
detections_top_value = info_item(info_bar, "◎", "Detections", "0", RED)

images_area = tk.Frame(main, bg=BG)
images_area.pack(fill="both", expand=True)

left_panel = make_panel(images_area)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

right_panel = make_panel(images_area)
right_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))

tk.Label(left_panel, text="Original Image", font=("Arial", 14, "bold"), fg=TEXT, bg=PANEL).pack(anchor="w", padx=18, pady=(15, 8))
tk.Label(right_panel, text="Detection Result", font=("Arial", 14, "bold"), fg=TEXT, bg=PANEL).pack(anchor="w", padx=18, pady=(15, 8))

original_viewer = ZoomCanvas(left_panel, "Original Image Preview")
result_viewer = ZoomCanvas(right_panel, "Detection Result Preview")

summary = make_panel(main)
summary.pack(fill="x", pady=(15, 0))

tk.Label(summary, text="DETECTION SUMMARY", font=("Arial", 12, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w", padx=18, pady=(14, 5))

summary_stats = tk.Frame(summary, bg=PANEL)
summary_stats.pack(fill="x", padx=12, pady=(0, 14))

total_value = stat_box(summary_stats, "◎", "Total Detections", RED)
avg_value = stat_box(summary_stats, "⌁", "Average Confidence", GREEN)

class_value = stat_box(summary_stats, "☷", "Class Detected", RED)
class_value.config(text="Oil Tank")

time_value = stat_box(summary_stats, "◷", "Processing Time", BLUE)
time_value.config(text="0.00 s")

root.mainloop()