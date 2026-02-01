import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from utils.logger import get_logger
from utils.config import DATA_DIR, MODELS_DIR, EMOTIONS

logger = get_logger("train_image")


def get_datasets(root: Path, img_size: int = 224):
    if not root.exists():
        raise FileNotFoundError(f"Image processed directory not found: {root}")
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(str(root), transform=train_tf)
    class_to_idx = ds.class_to_idx
    # Warn if classes don't match EMOTIONS
    ds_classes = list(class_to_idx.keys())
    missing = [e for e in EMOTIONS if e not in ds_classes]
    if missing:
        logger.warning(f"Image dataset missing emotion classes: {missing}. Found: {ds_classes}")

    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    # Set val transform explicitly
    val_ds.dataset.transform = val_tf

    return train_ds, val_ds, class_to_idx


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(train_ds, val_ds, num_classes: int, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader.dataset)
        acc = correct / total if total > 0 else 0.0
        logger.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")
        best_acc = max(best_acc, acc)

    return model, float(best_acc)


def save_artifacts(model, class_to_idx: dict, acc: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Model state
    model_path = out_dir / "resnet18.pt"
    torch.save(model.state_dict(), model_path)
    # Label map
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(class_to_idx, f)
    # Meta
    meta = {
        "architecture": "resnet18",
        "val_acc": acc,
        "emotions": EMOTIONS,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved image model artifacts to {out_dir}")


def main():
    img_proc_dir = DATA_DIR / "image" / "processed"
    try:
        train_ds, val_ds, class_to_idx = get_datasets(img_proc_dir)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        logger.info("Expected folder structure under data/image/processed with subfolders per emotion class containing images.")
        return

    logger.info(f"Loaded dataset with {len(train_ds)+len(val_ds)} images, classes={list(class_to_idx.keys())}")
    model, acc = train_model(train_ds, val_ds, num_classes=len(class_to_idx))

    out_dir = MODELS_DIR / "image_model"
    save_artifacts(model, class_to_idx, acc, out_dir)


if __name__ == "__main__":
    main()