import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# Custom dataset from csv
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

        filename = os.path.basename(row["Filename"])
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # crop to bounding box if available
        if self.use_bbox:
            try:
                x1, y1 = int(row["Upper left corner X"]), int(row["Upper left corner Y"])
                x2, y2 = int(row["Lower right corner X"]), int(row["Lower right corner Y"])
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            except Exception:
                pass

        label = self.label2idx[row["Annotation tag"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# CNN model
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes, img_size=128):
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
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# Training loop
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        correct, total, running_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("./model", exist_ok=True)
            torch.save(model.state_dict(), "./model/traffic_signs_best.pth")

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    return model



if __name__ == "__main__":
    # Dataset paths
    train_csv = "./1_Datasets/splits/train.csv"
    test_csv = "./1_Datasets/splits/test.csv"
    val_csv   = "./1_Datasets/splits/val.csv"
    img_dir   = "./1_Datasets/glaredataset"
    csv_file = "./1_Datasets/Classes.csv"

    # Categories
    df = pd.read_csv(csv_file)
    categories = sorted(df["Annotation tag"].unique())
    label2idx = {c: i for i, c in enumerate(categories)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(label2idx)

    print(f"Found {num_classes} categories: \n{categories}")

    # Image preprocessing
    img_size = 128
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load dataset
    train_dataset = TrafficSignsCSVDataset(train_csv, img_dir, label2idx, transform)
    val_dataset   = TrafficSignsCSVDataset(val_csv,   img_dir, label2idx, transform)
    test_dataset  = TrafficSignsCSVDataset(test_csv,  img_dir, label2idx, transform)


    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN(num_classes=num_classes, img_size=img_size)
    model = train_model(model, train_loader, val_loader, num_epochs=1000, lr=0.001, device=device)
