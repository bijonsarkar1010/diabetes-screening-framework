"""
Imaging-modality baseline for the multimodal diabetes screening pipeline.

Fine-tunes a pretrained ResNet18 on the APTOS 2019 Blindness Detection
dataset (retinal fundus images, 5-class diabetic retinopathy severity,
collapsed here to binary: no-DR vs. any-DR for a screening-style task) and
produces Grad-CAM saliency maps, corresponding to Framework Layer B
(Model & Explainability) and the imaging branch of Fig. 1 in the paper.

Expected input:
  data/aptos/train.csv          (columns: id_code, diagnosis)
  data/aptos/train_images/*.png

Run on a GPU (Colab/Kaggle) — this will be very slow on CPU.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix

from captum.attr import LayerGradCam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data/aptos"
IMG_DIR = os.path.join(DATA_DIR, "train_images")
LABELS_CSV = os.path.join(DATA_DIR, "train.csv")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # keep small for a first pass; raise once the pipeline works
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        # Binary screening label: 0 = no DR, 1 = any DR (diagnosis > 0)
        label = int(row["diagnosis"] > 0)
        return image, label


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def main():
    df = pd.read_csv(LABELS_CSV)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=(df["diagnosis"] > 0), random_state=RANDOM_STATE
    )

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = APTOSDataset(train_df, IMG_DIR, train_tf)
    test_ds = APTOSDataset(test_df, IMG_DIR, test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {running_loss/len(train_ds):.4f}")

    # Evaluation
    model.eval()
    all_labels, all_preds, all_proba = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_proba.extend(proba)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_proba),
        "sensitivity_recall": recall_score(all_labels, all_preds),
        "specificity": specificity_score(all_labels, all_preds),
    }
    print("=== Imaging model (ResNet18 on APTOS) ===")
    print(json.dumps(metrics, indent=2))

    with open("imaging_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), "imaging_model.pt")

    # Grad-CAM on a few test examples — Framework Layer B (imaging explainability)
    gradcam = LayerGradCam(model, model.layer4[-1])
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(DEVICE)
    attributions = gradcam.attribute(sample_images[:4], target=1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i in range(4):
        heatmap = attributions[i, 0].detach().cpu().numpy()
        axes[i].imshow(heatmap, cmap="jet")
        axes[i].set_title(f"Grad-CAM: sample {i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("imaging_gradcam_examples.png", dpi=150)
    print("Saved Grad-CAM examples to imaging_gradcam_examples.png")


if __name__ == "__main__":
    main()
