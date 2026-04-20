import os
import csv
from datetime import datetime

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


def save_config(run_dir, config_dict):
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")


def save_history_csv(run_dir, epoch_history, train_loss_history, val_loss_history,
                     train_iou_history, val_iou_history):
    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou"])

        for i in range(len(epoch_history)):
            writer.writerow([
                epoch_history[i],
                train_loss_history[i],
                val_loss_history[i],
                train_iou_history[i],
                val_iou_history[i],
            ])


def save_summary(run_dir, best_epoch, best_val_iou, final_train_loss, final_val_loss,
                 final_train_iou, final_val_iou):
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_iou={best_val_iou:.6f}\n")
        f.write(f"final_train_loss={final_train_loss:.6f}\n")
        f.write(f"final_val_loss={final_val_loss:.6f}\n")
        f.write(f"final_train_iou={final_train_iou:.6f}\n")
        f.write(f"final_val_iou={final_val_iou:.6f}\n")


def plot_metric(x, y_train, y_val, xlabel, ylabel, title, train_label, val_label, save_prefix):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_train, label=train_label, linestyle="-", linewidth=2)
    plt.plot(x, y_val, label=val_label, linestyle="--", linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path = f"{save_prefix}.png"
    eps_path = f"{save_prefix}.eps"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()

    print(f"Plot gespeichert als PNG: {png_path}")
    print(f"Plot gespeichert als EPS: {eps_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # Hyperparameter / Konfiguration
    # ---------------------------
    img_size = (256, 256)
    batch_size = 8
    val_ratio = 0.2
    num_workers = 2
    epochs = 20

    freeze_epochs = 5
    lr_phase1 = 1e-3
    lr_phase2 = 5e-4

    bce_weight = 0.3
    dice_weight = 0.7

    encoder_weights = "imagenet"

    # ---------------------------
    # Run-Ordner erstellen
    # ---------------------------
    results_dir = "results/supervised_training"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_lr1e3_to_5e4_freeze{freeze_epochs}"
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Speichere Ergebnisse in: {run_dir}")

    config = {
        "device": device,
        "img_size": img_size,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "num_workers": num_workers,
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "lr_phase1": lr_phase1,
        "lr_phase2": lr_phase2,
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
        "encoder_weights": encoder_weights,
    }
    save_config(run_dir, config)

    # ---------------------------
    # Daten
    # ---------------------------
    dataset = ISICDataset(
        "data/raw/images",
        "data/raw/masks_selected",
        img_size=img_size,
    )

    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ---------------------------
    # Modell
    # ---------------------------
    model = build_unet_resnet34(encoder_weights=encoder_weights).to(device)

    # Encoder zuerst einfrieren
    set_encoder_trainable(model, trainable=False)

    # Loss + Optimizer
    criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---------------------------
    # Historie
    # ---------------------------
    epoch_history = []
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []

    best_val_iou = -1.0
    best_epoch = -1

    # ---------------------------
    # Training
    # ---------------------------
    for epoch in range(1, epochs + 1):

        # Encoder nach freeze_epochs unfreezen
        if epoch == freeze_epochs + 1:
            print(f"Unfreezing encoder ab Epoche {epoch}")
            set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_phase2)

        # -------- TRAIN --------
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

        # -------- VALIDIERUNG --------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                val_iou += iou_from_logits(logits, masks)

        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        # Historie speichern
        epoch_history.append(epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_iou_history.append(train_iou)
        val_iou_history.append(val_iou)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        # Bestes Modell speichern
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch

            best_model_path = os.path.join(run_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "config": config,
            }, best_model_path)

            print(f"Bestes Modell gespeichert: {best_model_path}")

    # Letztes Modell speichern
    last_model_path = os.path.join(run_dir, "last_model.pth")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_iou": best_val_iou,
        "config": config,
    }, last_model_path)

    print(f"Letztes Modell gespeichert: {last_model_path}")

    # History CSV speichern
    save_history_csv(
        run_dir,
        epoch_history,
        train_loss_history,
        val_loss_history,
        train_iou_history,
        val_iou_history
    )

    # Summary speichern
    save_summary(
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_val_iou=best_val_iou,
        final_train_loss=train_loss_history[-1],
        final_val_loss=val_loss_history[-1],
        final_train_iou=train_iou_history[-1],
        final_val_iou=val_iou_history[-1],
    )

    # IoU-Plot speichern
    plot_metric(
        x=epoch_history,
        y_train=train_iou_history,
        y_val=val_iou_history,
        xlabel="Epoch",
        ylabel="IoU",
        title="Training and Validation IoU over Epochs",
        train_label="Train IoU",
        val_label="Validation IoU",
        save_prefix=os.path.join(run_dir, "iou_over_epochs")
    )

    # Loss-Plot speichern
    plot_metric(
        x=epoch_history,
        y_train=train_loss_history,
        y_val=val_loss_history,
        xlabel="Epoch",
        ylabel="Loss",
        title="Training and Validation Loss over Epochs",
        train_label="Train Loss",
        val_label="Validation Loss",
        save_prefix=os.path.join(run_dir, "loss_over_epochs")
    )

    print("\nTraining abgeschlossen.")
    print(f"Best Val IoU: {best_val_iou:.4f} in Epoche {best_epoch}")
    print(f"Alle Ergebnisse gespeichert in: {run_dir}")


if __name__ == "__main__":
    main()