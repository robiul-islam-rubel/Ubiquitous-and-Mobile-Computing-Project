import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# ======================================================
# 🔧 Model Builders
# ======================================================

def build_vgg16(num_classes, pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def build_resnet18(num_classes, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ======================================================
# ⚙️ Prediction Utilities
# ======================================================

def predict(image_path, model, transform, idx2label, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        class_idx = predicted.item()
        confidence = confidence.item()

    return idx2label[class_idx], confidence


def calculate_metrics(cm, idx2label):
    num_classes = len(idx2label)
    metrics = {}
    total = cm.sum().item()

    for i in range(num_classes):
        TP = cm[i, i].item()
        FN = cm[i, :].sum().item() - TP
        FP = cm[:, i].sum().item() - TP
        TN = total - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / total if total > 0 else 0.0

        metrics[idx2label[i]] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

    overall_precision = np.mean([m["precision"] for m in metrics.values()])
    overall_recall = np.mean([m["recall"] for m in metrics.values()])
    overall_accuracy = np.mean([m["accuracy"] for m in metrics.values()])

    overall = {
        "precision": overall_precision,
        "recall": overall_recall,
        "accuracy": overall_accuracy,
    }

    return metrics, overall


# ======================================================
# 🧪 Main Evaluation Pipeline
# ======================================================

if __name__ == "__main__":
    # --- Config ---
    model_name = "resnet"   # Choose: "vgg16" or "resnet"
    model_path = "./model/best_model.pth"
    csv_file   = "./1_Datasets/Classes.csv"
    test_csv   = "./1_Datasets/splits/test.csv"
    img_dir    = "./1_Datasets/glaredataset"
    save_csv   = "./Results/csv/glare_predictions.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Label mappings ---
    df_classes = pd.read_csv(csv_file)
    categories = sorted(df_classes["Annotation tag"].unique())
    label2idx = {c: i for i, c in enumerate(categories)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(label2idx)

    print(f"Loaded {num_classes} classes.")

    # --- Build model ---
    if model_name == "vgg16":
        model = build_vgg16(num_classes, pretrained=False)
    else:
        model = build_resnet18(num_classes, pretrained=False)

    # --- Load trained weights ---
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Preprocessing (same as training) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # important for VGG/ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Load test data ---
    df = pd.read_csv(test_csv)
    print(f"Total test images: {len(df)}")

    # --- Run inference ---
    y_true, y_pred = [], []
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = os.path.basename(row["Filename"])
        img_path = os.path.join(img_dir, filename)
        gt = row["Annotation tag"]

        pred, conf = predict(img_path, model, transform, idx2label, device)

        y_true.append(label2idx[gt])
        y_pred.append(label2idx[pred])

        results.append({
            "filename": filename,
            "ground_truth": gt,
            "prediction": pred,
            "confidence": conf
        })

    # --- Save predictions ---
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(save_csv, index=False)
    print(f"Predictions saved to {save_csv}")

    # --- Compute confusion matrix ---
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # --- Calculate metrics ---
    per_class, overall = calculate_metrics(cm, idx2label)

    print("\nPer-class metrics:")
    for cls, vals in per_class.items():
        print(f"{cls:20s} | Precision={vals['precision']:.2f}, Recall={vals['recall']:.2f}, Accuracy={vals['accuracy']:.2f}")

    print("\nOverall metrics:")
    print(f"Precision={overall['precision']:.2f}, Recall={overall['recall']:.2f}, Accuracy={overall['accuracy']:.2f}")
