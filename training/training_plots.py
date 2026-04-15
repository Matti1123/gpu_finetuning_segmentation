import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.U_net_resnet34 import build_unet_resnet34
from scripts.dataset import ISICDataset
from scripts.losses import BCEDiceLoss
from scripts.metrics import iou_from_logits


def set_encoder_trainable(model, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


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

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Modell
    model = build_unet_resnet34(encoder_weights="imagenet").to(device)

    # --- Transfer Learning
    freeze_epochs = 5
    set_encoder_trainable(model, trainable=False)

    # --- Loss + Optimizer
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Plot-Ausgabeordner
    results_dir = "results/supervised_training"
    os.makedirs(results_dir, exist_ok=True)

    # --- Historie für Plot
    train_iou_history = []
    val_iou_history = []
    epoch_history = []

    epochs = 20
    for epoch in range(1, epochs + 1):

        # Encoder nach freeze_epochs unfreezen
        if epoch == freeze_epochs + 1:
            print(f"Unfreezing encoder ab Epoche {epoch}")
            set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # ---- TRAIN
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_iou += iou_from_logits(logits.detach(), masks.detach())

        train_loss /= max(1, len(train_loader))
        train_iou /= max(1, len(train_loader))

        # ---- VALIDIERUNG
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item()
                val_iou += iou_from_logits(logits, masks)

        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        # --- Historie speichern
        epoch_history.append(epoch)
        train_iou_history.append(train_iou)
        val_iou_history.append(val_iou)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

    # --- Plot erstellen
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch_history,
        train_iou_history,
        label="Train IoU",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        epoch_history,
        val_iou_history,
        label="Validation IoU",
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path = os.path.join(results_dir, "iou_over_epochs.png")
    eps_path = os.path.join(results_dir, "iou_over_epochs.eps")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Plot gespeichert als PNG: {png_path}")
    print(f"Plot gespeichert als EPS: {eps_path}")


if __name__ == "__main__":
    main()