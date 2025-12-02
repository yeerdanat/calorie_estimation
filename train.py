from ultralytics import YOLO
def main():
    model = YOLO("yolov8s.pt")
    model.train(
        data="Food_Portion_Benchmark/FPB_Dataset/RGB/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="fpb_yolo",
        device = 0, 
        lr0=0.01, 
        patience = 20, 
        workers = 8
    )
if __name__ == "__main__":
    main()