import os
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
# #FOR CPU
# pip install torch torchvision torchaudio opencv-python pillow 
# #FOR GPU
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install opencv-python pillow
# ---------------- CONFIG ----------------
MODEL_PATH = "tumor_epoch10.pth"
IMAGE_PATH = "brain tumor.v1i.tensorflow/test/Te-me_0018_jpg.rf.bdeb58aa519d8b884aa2d69b91a4095d.jpg"
OUTPUT_DIR = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.5

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = 2 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)
image = Image.open(IMAGE_PATH).convert("RGB")
transform = torchvision.transforms.ToTensor()
img_tensor = transform(image).to(DEVICE)

with torch.no_grad():
    pred = model([img_tensor])[0]

boxes = pred["boxes"].cpu()
scores = pred["scores"].cpu()
labels = pred["labels"].cpu()

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, score, label in zip(boxes, scores, labels):
    if score < CONF_THRESH:
        continue
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 10), f"Tumor: {score:.2f}", fill="red", font=font)

output_path = os.path.join(OUTPUT_DIR, os.path.basename(IMAGE_PATH))
image.save(output_path)
print(f"✅ Saved result to: {output_path}")
