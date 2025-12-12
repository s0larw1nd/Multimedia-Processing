from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11l.pt")

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=2
    )