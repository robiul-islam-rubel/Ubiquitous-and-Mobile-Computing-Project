import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torchvision import transforms
import random
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Custom dataset 
class TrafficSignsCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, label2idx, transform=None, use_bbox=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.transform = transform
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # keep only filename, ignore subfolders in CSV
        filename = os.path.basename(row["Filename"])
        img_path = os.path.join(self.img_dir, filename)

        # load image
        image = Image.open(img_path).convert("RGB")

        # crop to bounding box if enabled
        if self.use_bbox:
            x1, y1 = max(0, int(row["Upper left corner X"])), max(0, int(row["Upper left corner Y"]))
            x2, y2 = int(row["Lower right corner X"]), int(row["Lower right corner Y"])
            if x2 > x1 and y2 > y1:  # only crop if valid
                image = image.crop((x1, y1, x2, y2))

        # map label
        label = self.label2idx[row["Annotation tag"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# Load CSV
df = pd.read_csv("../1_Datasets/Classes.csv")

# Count per class
class_counts = df["Annotation tag"].value_counts()

# Rare classes (<2 samples) → only training
rare_classes = class_counts[class_counts < 2].index
rare_df = df[df["Annotation tag"].isin(rare_classes)]
rest_df = df[~df["Annotation tag"].isin(rare_classes)]

print(f"⚠️ Classes with <2 samples forced into training only: {list(rare_classes)}")

# Initialize empty splits
train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if not rest_df.empty:
    try:
        # First split: Train vs Temp
        train_df, temp_df = train_test_split(
            rest_df,
            test_size=0.3,
            stratify=rest_df["Annotation tag"] if rest_df["Annotation tag"].nunique() > 1 else None,
            random_state=42
        )

        # Second split: Val vs Test
        if not temp_df.empty:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                stratify=temp_df["Annotation tag"] if temp_df["Annotation tag"].nunique() > 1 else None,
                random_state=42
            )

    except ValueError as e:
        print("Stratified split failed, falling back to random split:", e)
        # fallback: random split
        train_df, temp_df = train_test_split(rest_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Add rare classes back to training
train_df = pd.concat([train_df, rare_df], ignore_index=True)

# Save splits
os.makedirs("../1_Datasets/splits", exist_ok=True)
train_df.to_csv("../1_Datasets/splits/train.csv", index=False)
val_df.to_csv("../1_Datasets/splits/val.csv", index=False)
test_df.to_csv("../1_Datasets/splits/test.csv", index=False)

print(f"Split complete")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

print("\nTrain distribution:\n", train_df["Annotation tag"].value_counts())
print("\nVal distribution:\n", val_df["Annotation tag"].value_counts())
print("\nTest distribution:\n", test_df["Annotation tag"].value_counts())
