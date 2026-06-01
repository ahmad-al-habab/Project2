# gui_app3.py
# Oil Storage Tanks Segmentation + Cluster Density Analysis GUI
# Linux version - Reject button visibility fixed

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import os
import time
from datetime import datetime
import math

# =====================================================
# AUTO FIND MODEL - LINUX PATH
# =====================================================
PROJECT_DIR = Path(r"/home/alsarhan/PycharmProjects/oil_tanks_seg")

pt_files = list(PROJECT_DIR.rglob("best.pt"))

if len(pt_files) == 0:
    pt_files = list(PROJECT_DIR.rglob("last.pt"))

if len(pt_files) == 0:
    raise FileNotFoundError("No best.pt or last.pt found inside project")

print("\n========== FOUND MODELS ==========")
for i, pt in enumerate(pt_files):
    print(f"[{i}] {pt}")
print("==================================\n")

pt_files = sorted(pt_files, key=os.path.getmtime, reverse=True)
MODEL_PATH = str(pt_files[0])

print("========== USING MODEL ==========")
print("MODEL PATH:", MODEL_PATH)
print("ABS PATH:", os.path.abspath(MODEL_PATH))
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("=================================\n")

model = YOLO(MODEL_PATH)
print("LOADED MODEL:", model.ckpt_path)

device = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# =====================================================
# COLORS
# =====================================================
BG = "#0B1220"
SIDEBAR = "#111827"
CARD = "#162033"
CARD2 = "#0F172A"
BORDER = "#263449"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
BLUE = "#2563EB"
BLUE_HOVER = "#1D4ED8"
RED = "#EF4444"
ORANGE = "#F97316"
GREEN = "#22C55E"
WHITE = "#FFFFFF"
YELLOW = "#EAB308"

# =====================================================
# SETTINGS
# =====================================================
CLUSTER_DISTANCE = 180
MIN_CLUSTER_TANKS = 3
MEDIUM_CLUSTER_TANKS = 5
HIGH_CLUSTER_TANKS = 7

selected_image_path = None
original_tk = None
result_tk = None


# =====================================================
# IMAGE HELPERS
# =====================================================
def resize_for_display(img, max_w=520, max_h=340):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def bgr_to_tk(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(img_pil)


def show_image(label, img_bgr, side):
    global original_tk, result_tk

    display = resize_for_display(img_bgr)
    tk_img = bgr_to_tk(display)

    if side == "original":
        original_tk = tk_img
    else:
        result_tk = tk_img

    label.config(image=tk_img, text="")


# =====================================================
# CLUSTERING
# =====================================================
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_clusters(centers, max_distance=180):
    n = len(centers)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster = []

        while queue:
            current = queue.pop(0)
            cluster.append(current)

            for j in range(n):
                if not visited[j] and distance(centers[current], centers[j]) <= max_distance:
                    visited[j] = True
                    queue.append(j)

        clusters.append(cluster)

    return clusters


def classify_cluster(size):
    if size >= HIGH_CLUSTER_TANKS:
        return "HIGH DENSITY"
    elif size >= MEDIUM_CLUSTER_TANKS:
        return "MEDIUM DENSITY"
    elif size >= MIN_CLUSTER_TANKS:
        return "LOW DENSITY"
    else:
        return "ISOLATED"


# =====================================================
# DRAW RESULT + ANALYSIS
# =====================================================
def draw_segmentation_and_clusters(result, image):
    output = image.copy()

    detections = 0
    confidences = []
    centers = []
    boxes = []

    if result.masks is not None:
        masks = result.masks.xy
        detections = len(masks)

        if result.boxes is not None and result.boxes.conf is not None:
            confidences = result.boxes.conf.cpu().numpy().tolist()

        for i, mask in enumerate(masks):
            points = np.array(mask, dtype=np.int32)

            x, y, w, h = cv2.boundingRect(points)
            cx = x + w // 2
            cy = y + h // 2

            centers.append((cx, cy))
            boxes.append((x, y, w, h))

            overlay = output.copy()
            cv2.fillPoly(overlay, [points], color=(0, 0, 255))
            output = cv2.addWeighted(overlay, 0.22, output, 0.78, 0)

            cv2.polylines(output, [points], True, (0, 0, 255), 2)
            cv2.circle(output, (cx, cy), 4, (255, 255, 255), -1)

            if i < len(confidences):
                label = f"{confidences[i]:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.62
                thickness = 2

                text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_w, text_h = text_size

                label_x = x
                label_y = max(text_h + 10, y - 8)

                cv2.rectangle(
                    output,
                    (label_x, label_y - text_h - 8),
                    (label_x + text_w + 12, label_y + 5),
                    (15, 15, 15),
                    -1
                )

                cv2.rectangle(
                    output,
                    (label_x, label_y - text_h - 8),
                    (label_x + text_w + 12, label_y + 5),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    label,
                    (label_x + 6, label_y - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

    clusters = find_clusters(centers, CLUSTER_DISTANCE)

    valid_clusters = []
    largest_cluster = 0
    high_density_found = False

    for cluster_id, cluster in enumerate(clusters, start=1):
        size = len(cluster)
        density_label = classify_cluster(size)
        largest_cluster = max(largest_cluster, size)

        if size >= MIN_CLUSTER_TANKS:
            valid_clusters.append(cluster)

        xs, ys, xe, ye = [], [], [], []

        for idx in cluster:
            x, y, w, h = boxes[idx]
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        if len(xs) == 0:
            continue

        x1 = max(0, min(xs) - 18)
        y1 = max(0, min(ys) - 40)
        x2 = min(image.shape[1], max(xe) + 18)
        y2 = min(image.shape[0], max(ye) + 18)

        if size >= HIGH_CLUSTER_TANKS:
            color = (0, 0, 255)
            high_density_found = True
        elif size >= MEDIUM_CLUSTER_TANKS:
            color = (0, 165, 255)
        elif size >= MIN_CLUSTER_TANKS:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        if size >= MIN_CLUSTER_TANKS:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            text = f"Cluster {cluster_id} | {density_label} | {size} Tanks"

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 28)),
                (min(image.shape[1], x1 + 360), y1),
                color,
                -1
            )

            cv2.putText(
                output,
                text,
                (x1 + 8, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    avg_conf = sum(confidences) / len(confidences) if len(confidences) > 0 else 0
    cluster_count = len(valid_clusters)

    if high_density_found:
        decision = "High-density tank cluster detected. Human approval is required for monitoring."
        risk_level = "HIGH"
    elif largest_cluster >= MEDIUM_CLUSTER_TANKS:
        decision = "Medium-density tank cluster detected. Human review is recommended."
        risk_level = "MEDIUM"
    elif largest_cluster >= MIN_CLUSTER_TANKS:
        decision = "Low-density tank cluster detected. Continue observation only."
        risk_level = "LOW"
    else:
        decision = "No significant tank cluster detected. Isolated tanks are ignored."
        risk_level = "SAFE"

    analysis = {
        "detections": detections,
        "avg_conf": avg_conf,
        "cluster_count": cluster_count,
        "largest_cluster": largest_cluster,
        "decision": decision,
        "risk_level": risk_level
    }

    return output, analysis


# =====================================================
# BUTTON FUNCTIONS
# =====================================================
def select_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)

    if img is None:
        messagebox.showerror("Error", "Cannot read selected image")
        return

    selected_image_path = file_path
    h, w = img.shape[:2]

    show_image(original_image_label, img, "original")

    result_image_label.config(image="", text="Run Analysis to show result", fg=MUTED)

    image_name_value.config(text=os.path.basename(file_path))
    resolution_value.config(text=f"{w} x {h}")
    date_value.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    detections_value.config(text="--")
    total_det_value.config(text="--")
    avg_conf_value.config(text="--")
    clusters_value.config(text="--")
    largest_cluster_value.config(text="--")
    risk_value.config(text="--", fg=MUTED)
    decision_value.config(text="Waiting for analysis", fg=MUTED)
    time_value.config(text="--")

    approve_btn.config(state="disabled")
    reject_btn.config(state="disabled")
    status_value.config(text="Ready", fg=GREEN)


def run_analysis():
    if selected_image_path is None:
        messagebox.showwarning("Warning", "Please open an image first")
        return

    conf_value = confidence_slider.get() / 100

    status_value.config(text="Processing...", fg=YELLOW)
    decision_value.config(text="Analyzing tank density...", fg=YELLOW)
    root.update_idletasks()

    img = cv2.imread(selected_image_path)

    if img is None:
        messagebox.showerror("Error", "Cannot read image")
        return

    try:
        start_time = time.time()

        results = model.predict(
            source=selected_image_path,
            conf=conf_value,
            iou=0.5,
            imgsz=640,
            device=device,
            retina_masks=True,
            verbose=False
        )

        processing_time = time.time() - start_time
        result = results[0]

        output_img, analysis = draw_segmentation_and_clusters(result, img)

        show_image(result_image_label, output_img, "result")

        detections_value.config(text=str(analysis["detections"]))
        total_det_value.config(text=str(analysis["detections"]))
        avg_conf_value.config(text=f"{analysis['avg_conf']:.2f}" if analysis["avg_conf"] > 0 else "--")
        clusters_value.config(text=str(analysis["cluster_count"]))
        largest_cluster_value.config(text=str(analysis["largest_cluster"]))
        time_value.config(text=f"{processing_time:.2f} s")

        risk = analysis["risk_level"]

        if risk == "HIGH":
            risk_value.config(text="HIGH", fg=RED)
            decision_value.config(text=analysis["decision"], fg=RED)
            approve_btn.config(state="normal")
            reject_btn.config(state="normal")
        elif risk == "MEDIUM":
            risk_value.config(text="MEDIUM", fg=ORANGE)
            decision_value.config(text=analysis["decision"], fg=ORANGE)
            approve_btn.config(state="normal")
            reject_btn.config(state="normal")
        elif risk == "LOW":
            risk_value.config(text="LOW", fg=YELLOW)
            decision_value.config(text=analysis["decision"], fg=YELLOW)
            approve_btn.config(state="disabled")
            reject_btn.config(state="disabled")
        else:
            risk_value.config(text="SAFE", fg=GREEN)
            decision_value.config(text=analysis["decision"], fg=GREEN)
            approve_btn.config(state="disabled")
            reject_btn.config(state="disabled")

        status_value.config(text="Completed", fg=GREEN)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_value.config(text="Error", fg=RED)


def clear_all():
    global selected_image_path, original_tk, result_tk

    selected_image_path = None
    original_tk = None
    result_tk = None

    original_image_label.config(image="", text="Original Image", fg=MUTED)
    result_image_label.config(image="", text="Detection + Cluster Result", fg=MUTED)

    image_name_value.config(text="No image selected")
    resolution_value.config(text="--")
    date_value.config(text="--")
    detections_value.config(text="--")
    total_det_value.config(text="--")
    avg_conf_value.config(text="--")
    clusters_value.config(text="--")
    largest_cluster_value.config(text="--")
    risk_value.config(text="--", fg=MUTED)
    decision_value.config(text="Waiting for analysis", fg=MUTED)
    time_value.config(text="--")

    approve_btn.config(state="disabled")
    reject_btn.config(state="disabled")
    status_value.config(text="Ready", fg=GREEN)


def approve_monitoring():
    messagebox.showinfo(
        "Human Approval",
        "Operator approved the monitoring / inspection recommendation.\n\nNo autonomous action was performed."
    )
    decision_value.config(
        text="Human operator approved monitoring / inspection recommendation.",
        fg=GREEN
    )


def reject_monitoring():
    messagebox.showinfo(
        "Human Decision",
        "Operator rejected the monitoring / inspection recommendation.\n\nThe system will not proceed with any action."
    )
    decision_value.config(
        text="Human operator rejected the AI recommendation.",
        fg=RED
    )


def update_conf_label(value):
    confidence_value_label.config(text=f"{int(float(value)) / 100:.2f}")


# =====================================================
# UI HELPERS
# =====================================================
def make_button(parent, text, command, bg, active, height=2):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=WHITE,
        activebackground=active,
        activeforeground=WHITE,
        font=("Segoe UI", 10, "bold"),
        bd=0,
        height=height,
        cursor="hand2"
    )


def stat_card(parent, title, value, color):
    frame = tk.Frame(parent, bg=CARD2, highlightbackground=BORDER, highlightthickness=1)
    frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    tk.Label(
        frame,
        text=title,
        bg=CARD2,
        fg=MUTED,
        font=("Segoe UI", 8, "bold")
    ).pack(anchor="w", padx=10, pady=(8, 2))

    value_label = tk.Label(
        frame,
        text=value,
        bg=CARD2,
        fg=color,
        font=("Segoe UI", 11, "bold")
    )
    value_label.pack(anchor="w", padx=10, pady=(0, 8))

    return value_label


# =====================================================
# MAIN WINDOW
# =====================================================
root = tk.Tk()
root.title("Oil Tank Monitoring and Density Analysis System")
root.geometry("1366x768")
root.minsize(1100, 650)
root.configure(bg=BG)

try:
    root.state("zoomed")
except Exception:
    try:
        root.attributes("-zoomed", True)
    except Exception:
        pass

# =====================================================
# HEADER
# =====================================================
header = tk.Frame(root, bg="#08111F", height=58)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="◎  Oil Tank Monitoring System",
    bg="#08111F",
    fg=WHITE,
    font=("Segoe UI", 17, "bold")
).pack(anchor="w", padx=24, pady=(10, 0))

tk.Label(
    header,
    text="YOLOv8 Segmentation + Tank Cluster Density Analysis + Human-in-the-loop Decision Support",
    bg="#08111F",
    fg=MUTED,
    font=("Segoe UI", 10)
).pack(anchor="w", padx=76, pady=(0, 10))

# =====================================================
# BODY
# =====================================================
body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=16, pady=10)

# =====================================================
# SIDEBAR - FIXED FOR LINUX
# =====================================================
sidebar = tk.Frame(
    body,
    bg=SIDEBAR,
    width=235,
    highlightbackground=BORDER,
    highlightthickness=1
)
sidebar.pack(side="left", fill="y", padx=(0, 14))
sidebar.pack_propagate(False)

# Top area contains all controls and can shrink if Linux scaling is larger
top_sidebar = tk.Frame(sidebar, bg=SIDEBAR)
top_sidebar.pack(side="top", fill="both", expand=True)

# Bottom area is always visible; Reject button stays visible here
bottom_sidebar = tk.Frame(sidebar, bg=SIDEBAR)
bottom_sidebar.pack(side="bottom", fill="x", padx=16, pady=(4, 8))

tk.Label(
    top_sidebar,
    text="CONTROLS",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", padx=18, pady=(14, 8))

open_btn = make_button(top_sidebar, "📂  Open Image", select_image, BLUE, BLUE_HOVER, height=2)
open_btn.pack(fill="x", padx=18, pady=4)

clear_btn = make_button(top_sidebar, "🗑  Clear", clear_all, "#1F2937", "#374151", height=2)
clear_btn.pack(fill="x", padx=18, pady=4)

tk.Frame(top_sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=12)

tk.Label(
    top_sidebar,
    text="MODEL INFO",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", padx=18, pady=(0, 9))

tk.Label(
    top_sidebar,
    text="Model Name:",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(anchor="w", padx=18)

tk.Label(
    top_sidebar,
    text=os.path.basename(MODEL_PATH),
    bg=SIDEBAR,
    fg="#60A5FA",
    font=("Segoe UI", 9, "bold")
).pack(anchor="w", padx=18, pady=(2, 10))

tk.Label(
    top_sidebar,
    text="Class:",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(anchor="w", padx=18)

tk.Label(
    top_sidebar,
    text="●  Oil Tank",
    bg=SIDEBAR,
    fg=RED,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", padx=18, pady=(4, 12))

tk.Label(
    top_sidebar,
    text="Confidence Threshold",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(anchor="w", padx=18)

confidence_slider = tk.Scale(
    top_sidebar,
    from_=10,
    to=100,
    orient="horizontal",
    bg=SIDEBAR,
    fg=MUTED,
    troughcolor="#334155",
    highlightthickness=0,
    activebackground=BLUE,
    command=update_conf_label
)
confidence_slider.set(40)
confidence_slider.pack(fill="x", padx=18)

confidence_value_label = tk.Label(
    top_sidebar,
    text="0.40",
    bg=SIDEBAR,
    fg=TEXT,
    font=("Segoe UI", 10, "bold")
)
confidence_value_label.pack(anchor="center", pady=(0, 8))

tk.Label(
    top_sidebar,
    text="Cluster Distance: 180 px",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 8)
).pack(anchor="w", padx=18)

tk.Label(
    top_sidebar,
    text="Minimum Cluster: 3 tanks",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 8)
).pack(anchor="w", padx=18, pady=(2, 8))

run_btn = make_button(top_sidebar, "▶  Run Analysis", run_analysis, BLUE, BLUE_HOVER, height=2)
run_btn.pack(fill="x", padx=18, pady=4)

# Bottom fixed buttons section
tk.Frame(bottom_sidebar, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

tk.Label(
    bottom_sidebar,
    text="HUMAN-IN-THE-LOOP",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", pady=(0, 8))

approve_btn = make_button(bottom_sidebar, "✅  Approve Inspection", approve_monitoring, GREEN, "#16A34A", height=2)
approve_btn.pack(fill="x", pady=4)
approve_btn.config(state="disabled")

reject_btn = make_button(bottom_sidebar, "❌  Reject", reject_monitoring, RED, "#DC2626", height=2)
reject_btn.pack(fill="x", pady=4)
reject_btn.config(state="disabled")

tk.Frame(bottom_sidebar, bg=BORDER, height=1).pack(fill="x", pady=(10, 8))

tk.Label(
    bottom_sidebar,
    text="STATUS",
    bg=SIDEBAR,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w")

status_value = tk.Label(
    bottom_sidebar,
    text="Ready",
    bg=SIDEBAR,
    fg=GREEN,
    font=("Segoe UI", 10, "bold")
)
status_value.pack(anchor="w", pady=(5, 0))

# =====================================================
# MAIN CONTENT
# =====================================================
content = tk.Frame(body, bg=BG)
content.pack(side="right", fill="both", expand=True)

# =====================================================
# INFO BAR
# =====================================================
info_bar = tk.Frame(
    content,
    bg=CARD,
    height=68,
    highlightbackground=BORDER,
    highlightthickness=1
)
info_bar.pack(fill="x")
info_bar.pack_propagate(False)

info1 = tk.Frame(info_bar, bg=CARD)
info1.pack(side="left", expand=True, fill="both", padx=18)

tk.Label(info1, text="Image Name", bg=CARD, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 0))
image_name_value = tk.Label(info1, text="No image selected", bg=CARD, fg="#60A5FA", font=("Segoe UI", 9))
image_name_value.pack(anchor="w")

info2 = tk.Frame(info_bar, bg=CARD)
info2.pack(side="left", expand=True, fill="both")

tk.Label(info2, text="Resolution", bg=CARD, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 0))
resolution_value = tk.Label(info2, text="--", bg=CARD, fg=TEXT, font=("Segoe UI", 9))
resolution_value.pack(anchor="w")

info3 = tk.Frame(info_bar, bg=CARD)
info3.pack(side="left", expand=True, fill="both")

tk.Label(info3, text="Date", bg=CARD, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 0))
date_value = tk.Label(info3, text="--", bg=CARD, fg="#60A5FA", font=("Segoe UI", 9))
date_value.pack(anchor="w")

info4 = tk.Frame(info_bar, bg=CARD)
info4.pack(side="left", expand=True, fill="both")

tk.Label(info4, text="Detections", bg=CARD, fg=RED, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 0))
detections_value = tk.Label(info4, text="--", bg=CARD, fg=RED, font=("Segoe UI", 15, "bold"))
detections_value.pack(anchor="w")

# =====================================================
# IMAGE AREA
# =====================================================
image_area = tk.Frame(content, bg=BG)
image_area.pack(fill="both", expand=False, pady=10)
image_area.configure(height=430)
image_area.pack_propagate(False)

left_card = tk.Frame(
    image_area,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)
left_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

right_card = tk.Frame(
    image_area,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)
right_card.pack(side="right", fill="both", expand=True, padx=(6, 0))

tk.Label(
    left_card,
    text="Original Image",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 11, "bold")
).pack(anchor="w", padx=16, pady=(10, 6))

tk.Label(
    right_card,
    text="Segmentation + Cluster Analysis",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 11, "bold")
).pack(anchor="w", padx=16, pady=(10, 6))

original_image_label = tk.Label(
    left_card,
    text="Original Image",
    bg="#020617",
    fg=MUTED,
    font=("Segoe UI", 14)
)
original_image_label.pack(fill="both", expand=False, padx=10, pady=(0, 10))

result_image_label = tk.Label(
    right_card,
    text="Detection + Cluster Result",
    bg="#020617",
    fg=MUTED,
    font=("Segoe UI", 14)
)
result_image_label.pack(fill="both", expand=False, padx=10, pady=(0, 10))

# =====================================================
# SUMMARY
# =====================================================
summary = tk.Frame(
    content,
    bg=CARD,
    height=105,
    highlightbackground=BORDER,
    highlightthickness=1
)
summary.pack(fill="x")
summary.pack_propagate(False)

tk.Label(
    summary,
    text="ANALYSIS SUMMARY",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 9, "bold")
).pack(anchor="w", padx=18, pady=(6, 0))

summary_cards = tk.Frame(summary, bg=CARD)
summary_cards.pack(fill="x", padx=10, pady=3)

total_det_value = stat_card(summary_cards, "Total Tanks", "--", RED)
avg_conf_value = stat_card(summary_cards, "Avg Confidence", "--", GREEN)
clusters_value = stat_card(summary_cards, "Clusters", "--", "#60A5FA")
largest_cluster_value = stat_card(summary_cards, "Largest Cluster", "--", ORANGE)
risk_value = stat_card(summary_cards, "Density Level", "--", MUTED)
time_value = stat_card(summary_cards, "Processing Time", "--", "#60A5FA")

decision_frame = tk.Frame(summary, bg=CARD)
decision_frame.pack(fill="x", padx=18)

tk.Label(
    decision_frame,
    text="AI Recommendation:",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 9, "bold")
).pack(side="left")

decision_value = tk.Label(
    decision_frame,
    text="Waiting for analysis",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 10, "bold")
)
decision_value.pack(side="left", padx=10)

# =====================================================
# RUN
# =====================================================
root.mainloop()
