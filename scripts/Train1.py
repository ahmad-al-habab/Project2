from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        patience=20,
        device=0,
        workers=0,
        lr0=0.001
    )

if __name__ == "__main__":
    main()