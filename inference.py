# Importamos librerias
from ultralytics import YOLO
import cv2
import math
import os

# Modelo
model = YOLO('modelos/best.onnx')

imgpath = 'data/img'

# List
images = []
clases = []
lista = os.listdir(imgpath)

# Leemos las imagenes del DB
for lis in lista:
    # Leemos las imagenes de los rostros
    imgdb = cv2.imread(f'{imgpath}/{lis}')
    # Almacenamos imagen
    images.append(imgdb)
    # Almacenamos nombre
    clases.append(os.path.splitext(lis)[0])

count = 0

numImganes = len(images)

print(f'Numero imagenes: {numImganes}')

# Clases
clsName = ["invoice"]

# Inference
while(count < numImganes):

    img = images[count]
    imgCopy = img.copy()

    # Yolo | AntiSpoof
    results = model(img, imgsz = 640)
    for res in results:
        # Box
        boxes = res.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Error < 0
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0

            # Class
            cls = int(box.cls[0])

            # Confidence
            conf = math.ceil(box.conf[0])
            print(f"Clase: {cls} Confidence: {conf}")

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'{clsName[cls]} {int(conf * 100)}%', (x1, y1 - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Show
    cv2.imshow("Invoice", img)

    # Close
    t = cv2.waitKey(0)
    if t == 27:
        break

cv2.destroyAllWindows()