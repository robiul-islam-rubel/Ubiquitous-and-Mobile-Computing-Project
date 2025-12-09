import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import glob
import os
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json


# --- CNN model (same as training) ---
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (128 // 8) * (128 // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Load model and mappings ---
checkpoint = torch.load(f"./model/best_model.pth", map_location="cpu")
label2idx = checkpoint["label2idx"]
idx2label = checkpoint["idx2label"]

model = TrafficSignCNN(num_classes=len(label2idx))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Preprocessing (same as training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Prediction Function ---
def predict(image_path, bbox=None):
    image = Image.open(image_path).convert("RGB")

    # crop if bounding box is provided
    if bbox:
        x1, y1, x2, y2 = bbox
        image = image.crop((x1, y1, x2, y2))

    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(image)             
        probs = F.softmax(outputs, dim=1)   
        confidence, predicted = torch.max(probs, 1)  
        class_idx = predicted.item()
        confidence = confidence.item()     

    return idx2label[class_idx], confidence

def calculate_metrics(metrics):
    per_class = {}
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0

    for cls, vals in metrics.items():
        TP, FP, FN, TN = vals["TP"], vals["FP"], vals["FN"], vals["TN"]

        # Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Overall metrics
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

    overall = {
        "precision": overall_precision,
        "recall": overall_recall,
        "accuracy": overall_accuracy
    }

    return per_class, overall



if __name__=="__main__":
    
    # Load CSV
    df = pd.read_csv("./1_Datasets/splits/test.csv")
    print(len(df))
    # Collect all ground truths and predictions
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        filename = os.path.basename(row["Filename"])
        img_path = os.path.join("./1_Datasets/glaredataset", filename)
        gt = row["Annotation tag"]
        pred, conf = predict(img_path)
        print(f"GT: {gt}, Pred: {pred}, Confidence: {conf}")

        y_true.append(label2idx[gt])
        y_pred.append(label2idx[pred])

    # Build confusion matrix
    num_classes = len(label2idx)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Compute TP, FP, FN, TN per class
    metrics = {}
    total = cm.sum().item()
    for i in range(num_classes):
        TP = cm[i, i].item()
        FN = cm[i, :].sum().item() - TP
        FP = cm[:, i].sum().item() - TP
        TN = total - (TP + FP + FN)
        metrics[idx2label[i]] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

    print(metrics)
    per_class, overall = calculate_metrics(metrics)

    print("Per-class metrics:")
    for cls, vals in per_class.items():
        print(f"{cls}: Precision={vals['precision']:.2f}, Recall={vals['recall']:.2f}, Accuracy={vals['accuracy']:.2f}")

    print("\nOverall metrics:")
    print(f"Precision={overall['precision']:.2f}, Recall={overall['recall']:.2f}, Accuracy={overall['accuracy']:.2f}")


    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = os.path.basename(row["Filename"])
        img_path = os.path.join("./1_Datasets/glaredataset", filename)
        gt = row["Annotation tag"]
        pred, conf = predict(img_path)

        results.append({
            "filename": filename,
            "ground_truth": gt,
            "prediction": pred,
            "confidence": conf
        })

    pd.DataFrame(results).to_csv("./Results/csv/glare_predictions.csv", index=False)



