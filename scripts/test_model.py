from ultralytics import YOLO

model = YOLO("runs/segment/train-8/weights/best.pt")

results = model(
    source="dataset_tiled_yolo/images/test",  # أو صورة وحدة
    conf=0.4,
    iou=0.5,
    retina_masks=True,
    save=True
)