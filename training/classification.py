# training/train_classifier_from_unet_encoder.py

import os
import csv

from timm.models import checkpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.U_net_resnet34 import build_unet_resnet34
from models.resnet_classifier import ResNet34ClassifierFromUNet
from scripts.classification_dataset import ISICClassificationDataset


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        val_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def save_plots(log_path, save_dir):
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_acc"]))
            val_accs.append(float(row["val_acc"]))

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=300)
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "data/classification_dataset/ISIC2018_Task3_Training_Input"
    csv_path = "data/classification_dataset/ISIC2018_Task3_Training_GroundTruth.csv"
    unet_checkpoint_path = "runs/exp_first/best.pt"

    save_dir = "results/classifier_exp_2"
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train_log.csv")

    batch_size = 16
    epochs = 20
    freeze_epochs = 10

    lr_head = 1e-3
    lr_finetune = 1e-6
    val_ratio = 0.2

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ISICClassificationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=transform
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    # U-Net bauen
    unet = build_unet_resnet34()
    checkpoint = torch.load(unet_checkpoint_path, map_location=device)

    if "model_state" in checkpoint:
        unet.load_state_dict(checkpoint["model_state"])
    else:
        unet.load_state_dict(checkpoint)

    model = ResNet34ClassifierFromUNet(
        unet_model=unet,
        num_classes=7,
        freeze_encoder=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head
    )

    best_val_acc = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
            "encoder_frozen"
        ])

    for epoch in range(1, epochs + 1):

        if epoch == freeze_epochs + 1:
            print("Encoder wird freigegeben.")

            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr_finetune
            )

        encoder_frozen = not any(
            param.requires_grad for param in model.encoder.parameters()
        )

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                "loss": loss.item(),
                "acc": correct / total
            })

        train_loss /= len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                encoder_frozen
            ])

        save_plots(log_path, save_dir)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_classifier.pt")
            )

            print(f"Bestes Modell gespeichert: Val Acc = {best_val_acc:.4f}")

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "last_classifier.pt")
    )

    print("Training abgeschlossen.")
    print(f"Beste Validation Accuracy: {best_val_acc:.4f}")
    print(f"Log gespeichert unter: {log_path}")


if __name__ == "__main__":
    train()