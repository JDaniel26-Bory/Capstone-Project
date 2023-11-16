# Libraries
import os
import cv2
import torch
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

# CONFIG SAM
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# CheckPoint Weights
check_point_pathSAM = 'sam/sam_vit_h_4b8939.pth'

# Model SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=check_point_pathSAM).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
sam_predictor = SamPredictor(sam)


# CONFIG DINO
home = os.getcwd()

# Config Path
config_pathDINO = os.path.join(home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# CheckPoint Weights
check_point_pathDINO = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'

# Model
modelDINO = load_model(config_pathDINO, check_point_pathDINO)

# Prompt
text_prompt = 'ticket'
box_threshold = 0.35
text_threshold = 0.25

# Variables
con = 0
OutFolderPath = 'debug/'
classID = 1

# Function Draw
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# PROCESS
img = cv2.imread('03.jpeg')

# Transform
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Convert img to PIL object
img_source = Image.fromarray(img).convert("RGB")
# Convert PIL image onject to transform object
img_transform, _ = transform(img_source, None)

# Predict
boxes, logits, phrases = predict(
    model=modelDINO,
    image=img_transform,
    caption=text_prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
    device='cpu')

# SAM inference
sam_predictor.set_image(image=img)
H, W, c = img.shape

# box: normalized box xywh -> unnormalized xyxy
boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, img.shape[:2]).to(DEVICE)
masks, scores, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False)


# Annotated
annotated_img = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
annotated_frame_with_mask = show_mask(masks[0][0], annotated_img)

# Display
out_frame = cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB)
cv2.imshow('DINO', out_frame)

# Save Image With Draw
cv2.imwrite(f"{OutFolderPath}/mask.jpg", out_frame)
# cv2.imwrite(f"{OutFolderPath}/Binarymask.jpg", masked)
cv2.waitKey(0)