# Production-ready Tumor Detection (Faster R-CNN)
# Dataset structure:
# brain tumor.v1i.tensorflow/
# ├── train/_annotations.csv
# ├── valid/_annotations.csv
# └── test/_annotations.csv
# Each CSV: filename,width,height,class,xmin,ymin,xmax,ymax

import os
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------- Dataset ----------------
class RoboflowCSVDataset(Dataset):
    def __init__(self, csv_path, images_root, transforms=None):
        import csv
        self.images_root = Path(images_root)
        self.transforms = transforms
        self.records = defaultdict(list)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = r['filename']
                self.records[fname].append([
                    float(r['xmin']), float(r['ymin']),
                    float(r['xmax']), float(r['ymax']),
                    1  # tumor label
                ])
        self.filenames = sorted(self.records.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = self.images_root / fname
        img = Image.open(img_path).convert('RGB')

        boxes = torch.as_tensor([b[:4] for b in self.records[fname]], dtype=torch.float32)
        labels = torch.as_tensor([b[4] for b in self.records[fname]], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

# ---------------- Transforms ----------------
class TrainTransforms:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
            w = image.shape[2]
            target['boxes'][:, [0, 2]] = w - target['boxes'][:, [2, 0]]
        return image, target

class EvalTransforms:
    def __call__(self, image, target):
        return F.to_tensor(image), target

# ---------------- Model ----------------
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------- Utils ----------------
def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch} Iter {i+1}/{len(loader)} Loss: {loss.item():.4f}")
    print(f"Epoch {epoch} Avg Loss: {total_loss/len(loader):.4f}")

def evaluate_simple(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                gt = tgt['boxes'].cpu().numpy()
                pr = out['boxes'].cpu().numpy()
                if len(gt) == 0: continue
                for p in pr:
                    iou = bbox_iou_batch(p, gt)
                    if len(iou) and iou.max() > 0.5: tp += 1
                    else: fp += 1
                fn += max(0, len(gt) - tp)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

def bbox_iou_batch(box, boxes):
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    areaA = (box[2]-box[0]) * (box[3]-box[1])
    areaB = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (areaA + areaB - inter + 1e-8)

# ---------------- Train Script ----------------
def main():
    data_dir = r"C:\\Users\\Hp\\work\\personal\\cancer identification ai\\Tumora\\brain tumor.v1i.tensorflow"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_csv = Path(data_dir)/'train'/'_annotations.csv'
    valid_csv = Path(data_dir)/'valid'/'_annotations.csv'

    train_loader = DataLoader(RoboflowCSVDataset(train_csv, Path(data_dir)/'train', TrainTransforms()),
                              batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(RoboflowCSVDataset(valid_csv, Path(data_dir)/'valid', EvalTransforms()),
                            batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = get_model(2).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    epochs = 10
    for epoch in range(1, epochs+1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        evaluate_simple(model, val_loader, device)
        torch.save(model.state_dict(), f"tumor_epoch{epoch}.pth")

if __name__ == '__main__':
    main()
