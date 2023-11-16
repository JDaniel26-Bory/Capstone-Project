# Libraries
from ultralytics import YOLO

# Model
model = YOLO('modelos/yolov8l.pt')

def main():
    # Train
    model.train(data = 'data/splitData/Dataset.yaml', epochs = 30, batch = 4, imgsz = 640) # HiperParametro de entrenamiento

if __name__ == '__main__':
    main()