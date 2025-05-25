import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import json
import os
from rfdetr.models.rf_detr import build_model  # Adjust if your path is different
from rfdetr.datasets.coco import make_coco_transforms  # Adjust if needed
from rfdetr.util.misc import nested_tensor_from_tensor_list

# ====== CONFIG ======
IMAGE_PATH = "C:/Users/karth/Documents/TattooTest-1/test/19_1_i1_JPG_jpg.rf.874d0f6a6db7660c0eda57bb50bc656e.jpg"
CHECKPOINT_PATH = "checkpoints/model_epoch_3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.5
OUTPUT_PATH = "inference_output.png"

# ====== LOAD MODEL ======
print("🔧 Loading RF-DETR model...")
model, _, _ = build_model(argparse.Namespace(backbone="facebook/dinov2-base"))
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# ====== LOAD IMAGE ======
print(f"🖼️ Running inference on: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tensor_image = transform(image).unsqueeze(0).to(DEVICE)
sample = nested_tensor_from_tensor_list([tensor_image])

# ====== INFERENCE ======
with torch.no_grad():
    outputs = model(sample)

# ====== POSTPROCESSING ======
logits = outputs['pred_logits'][0]
boxes = outputs['pred_boxes'][0]
probs = logits.sigmoid().max(-1)
scores = probs.values
labels = probs.indices

keep = scores > CONF_THRESHOLD
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

# ====== DRAW BOXES ======
draw = ImageDraw.Draw(image)
for box, label, score in zip(boxes, labels, scores):
    cx, cy, w, h = box
    cx *= image.width
    cy *= image.height
    w *= image.width
    h *= image.height
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2
    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    draw.text((x0, y0), f"{label.item()} ({score:.2f})", fill="red")

image.save(OUTPUT_PATH)
print(f"Inference completed. Saved to: {OUTPUT_PATH}")

