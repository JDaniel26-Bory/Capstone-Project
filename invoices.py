# Libraries
import os
import cv2
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from segment_anything import SamPredictor
from time import time

home = os.getcwd()
# Config Path
config_path = os.path.join(home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# CheckPoint Weights
check_point_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'

# Model
model = load_model(config_path, check_point_path)

# Prompt
text_prompt = 'ticket'
box_threshold = 0.35
text_threshold = 0.25

img = cv2.imread('data/img/01.jpg')


# Transform
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# clsName = ["Ticket"]

# Convert img to PIL object
img_source = Image.fromarray(img).convert("RGB")

# Convert PIL image onject to transform object
img_transform, _ = transform(img_source, None)

# Predict
boxes, logits, phrases = predict(
    model=model,
    image=img_transform,
    caption=text_prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
    device='cpu')

# detectiont = predict(model=model,
#         image=img_transform,
#         caption=text_prompt,
#         box_threshold=box_threshold,
#         text_threshold=text_threshold,
#         device='cpu')

alto, ancho, c = img.shape



cv2.circle(img, (y1, y2), 5, (255,0,0), -1)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

print(f'Boxes: {boxes}, Alto: {alto}, Ancho: {ancho}')

# Annotated
# annotated_img = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
# annotated_img = annotate(image_source=img, detections=detectiont, labels=clsName)


# display the output
# out_frame = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)


cv2.imshow('DINO', img)

cv2.waitKey(0)