import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import segmentation_models_pytorch as smp
from scripts.dataset import ISICDataset

# Metrics / Loss

def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice Loss für binäre Segmentierung.
    logits: [B,1,H,W] (roh, ohne Sigmoid)
    targets: [B,1,H,W] (0/1)
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

@torch.no_grad()
def iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())

def set_encoder_trainable(model, trainable: bool):
    # smp.Unet hat model.encoder
    for p in model.encoder.parameters():
        p.requires_grad = trainable

# Training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Daten
    dataset = ISICDataset(
        "data/raw/images",
        "data/raw/masks_selected",
        img_size=(256, 256),
    )

    # --- Split (80/20)
    val_ratio = 0.2
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # --- Modell: U-Net mit pretrained Encoder (Transfer Learning)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None, 
    ).to(device)



    # Transfer Learning: Encoder zuerst einfrieren
    freeze_epochs = 5 # wirab der 5 Epoche wird der Encoder mittrainiert (geunfreezt)
    set_encoder_trainable(model, trainable=False)  # wir lassen das Freezen weg

    # --- Loss + Optimizer
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Output
    out_dir = "runs/exp_0_3_BCE_dice_0_7"  # da wir hier nicht einfrieren, anderer Ordner
    os.makedirs(out_dir, exist_ok=True)
    best_val_iou = -1.0

    epochs = 20
    for epoch in range(1, epochs + 1):

        # Encoder nach freeze_epochs "unfreezen"
        if epoch == freeze_epochs + 1:
            print(f"Unfreezing encoder ab Epoche {epoch}")
            set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # kleinerer LR beim Unfreezen

        # ---- TRAIN
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)  # [B,1,H,W]
                loss = 0.3 * bce(logits, masks) + 0.7 * dice_loss_from_logits(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_iou  += iou_from_logits(logits.detach(), masks.detach())

        train_loss /= max(1, len(train_loader))
        train_iou  /= max(1, len(train_loader))

        # ---- VAL
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = 0.3 * bce(logits, masks) + 0.7 * dice_loss_from_logits(logits, masks)

                val_loss += loss.item()
                val_iou  += iou_from_logits(logits, masks)

        val_loss /= max(1, len(val_loader))
        val_iou  /= max(1, len(val_loader))

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_iou={train_iou:.4f} | val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # ---- Checkpoint speichern (bestes Modell)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.path.join(out_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
            }, ckpt_path)
            print(f"✅ Bestes Modell gespeichert: {ckpt_path} (val_iou={best_val_iou:.4f})")

if __name__ == "__main__":
    main()