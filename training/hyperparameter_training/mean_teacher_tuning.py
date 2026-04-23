import os
import math
import csv
import random
from datetime import datetime
from itertools import cycle

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.U_net_resnet34 import build_unet_resnet34
from scripts.dataset import ISICDataset
from scripts.unlabeled_dataset import ISICUnlabeledDataset
from scripts.losses import BCEDiceLoss, consistency_loss_from_logits
from scripts.metrics import iou_from_logits
from scripts.ema import copy_student_to_teacher, update_teacher


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    val_iou = 0.0

    for images, masks in tqdm(val_loader, desc="[val]", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        val_loss += loss.item()
        val_iou += iou_from_logits(logits, masks)

    val_loss /= max(1, len(val_loader))
    val_iou /= max(1, len(val_loader))
    return val_loss, val_iou


def build_splits(full_dataset, val_fraction=0.2, label_fraction=0.2, seed=42):
    n_total = len(full_dataset)
    indices = list(range(n_total))

    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(n_total * val_fraction)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    n_labeled = max(1, int(len(train_indices) * label_fraction))
    labeled_indices = train_indices[:n_labeled]
    unlabeled_indices = train_indices[n_labeled:]

    return labeled_indices, unlabeled_indices, val_indices


def set_encoder_trainable(model, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable

def save_config(run_dir, config_dict):
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")


def save_history_csv(
    run_dir,
    epoch_history,
    train_total_loss_history,
    train_sup_loss_history,
    train_unsup_loss_history,
    train_iou_history,
    lambda_u_history,
    val_student_loss_history,
    val_student_iou_history,
    val_teacher_loss_history,
    val_teacher_iou_history,
):
    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_sup_loss",
            "train_unsup_loss",
            "train_iou",
            "lambda_u",
            "val_student_loss",
            "val_student_iou",
            "val_teacher_loss",
            "val_teacher_iou",
        ])

        for i in range(len(epoch_history)):
            writer.writerow([
                epoch_history[i],
                train_total_loss_history[i],
                train_sup_loss_history[i],
                train_unsup_loss_history[i],
                train_iou_history[i],
                lambda_u_history[i],
                val_student_loss_history[i],
                val_student_iou_history[i],
                val_teacher_loss_history[i],
                val_teacher_iou_history[i],
            ])


def save_summary(
    run_dir,
    best_epoch_teacher,
    best_val_iou_teacher,
    final_train_total_loss,
    final_train_sup_loss,
    final_train_unsup_loss,
    final_train_iou,
    final_val_student_loss,
    final_val_student_iou,
    final_val_teacher_loss,
    final_val_teacher_iou,
):
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch_teacher={best_epoch_teacher}\n")
        f.write(f"best_val_iou_teacher={best_val_iou_teacher:.6f}\n")
        f.write(f"final_train_total_loss={final_train_total_loss:.6f}\n")
        f.write(f"final_train_sup_loss={final_train_sup_loss:.6f}\n")
        f.write(f"final_train_unsup_loss={final_train_unsup_loss:.6f}\n")
        f.write(f"final_train_iou={final_train_iou:.6f}\n")
        f.write(f"final_val_student_loss={final_val_student_loss:.6f}\n")
        f.write(f"final_val_student_iou={final_val_student_iou:.6f}\n")
        f.write(f"final_val_teacher_loss={final_val_teacher_loss:.6f}\n")
        f.write(f"final_val_teacher_iou={final_val_teacher_iou:.6f}\n")


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

    # -------------------------
    # Hyperparameter
    # -------------------------
    img_size = (256, 256)
    val_fraction = 0.2
    label_fraction = 0.2
    seed = 42

    batch_size_labeled = 2
    batch_size_unlabeled = 10
    batch_size_val = 8
    num_workers = 2

    epochs = 15
    freeze_epochs = 5
    lr_phase1 = 1e-3
    lr_phase2 = 5e-4
    ema_decay = 0.99

    unsup_weight_max = 0.1
    rampup_epochs = 12
    confidence_threshold = 0.8

    bce_weight = 0.3
    dice_weight = 0.7

    encoder_weights = "imagenet"

    # -------------------------
    # Run-Ordner
    # -------------------------
    results_dir = "results/semi_supervised_training"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{timestamp}_mean_teacher_"
        f"split{int(label_fraction*100)}_thr{str(confidence_threshold).replace('.', '')}"
    )
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Speichere Ergebnisse in: {run_dir}")

    # -------------------------
    # Dataset laden
    # -------------------------
    full_dataset = ISICDataset(
        images_dir="data/raw/images",
        masks_dir="data/raw/masks_selected",
        img_size=img_size,
    )

    labeled_indices, unlabeled_indices, val_indices = build_splits(
        full_dataset=full_dataset,
        val_fraction=val_fraction,
        label_fraction=label_fraction,
        seed=seed,
    )

    print(f"Full dataset:     {len(full_dataset)}")
    print(f"Labeled train:    {len(labeled_indices)}")
    print(f"Unlabeled train:  {len(unlabeled_indices)}")
    print(f"Validation:       {len(val_indices)}")

    config = {
        "device": device,
        "model_type": "mean_teacher_unet_resnet34",
        "img_size": img_size,
        "val_fraction": val_fraction,
        "label_fraction": label_fraction,
        "seed": seed,
        "batch_size_labeled": batch_size_labeled,
        "batch_size_unlabeled": batch_size_unlabeled,
        "batch_size_val": batch_size_val,
        "num_workers": num_workers,
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "lr_phase1": lr_phase1,
        "lr_phase2": lr_phase2,
        "ema_decay": ema_decay,
        "unsup_weight_max": unsup_weight_max,
        "rampup_epochs": rampup_epochs,
        "confidence_threshold": confidence_threshold,
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
        "encoder_weights": encoder_weights,
        "num_labeled_samples": len(labeled_indices),
        "num_unlabeled_samples": len(unlabeled_indices),
        "num_val_samples": len(val_indices),
    }
    save_config(run_dir, config)

    # -------------------------
    # Subsets
    # -------------------------
    labeled_dataset = Subset(full_dataset, labeled_indices)
    val_dataset = Subset(full_dataset, val_indices)

    unlabeled_image_names = [
        full_dataset.image_dict[full_dataset.paired_keys[i]] for i in unlabeled_indices
    ]

    unlabeled_dataset = ISICUnlabeledDataset(
        images_dir="data/raw/images",
        img_size=img_size,
        image_list=unlabeled_image_names,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size_labeled,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size_unlabeled,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Modelle
    # -------------------------
    student_model = build_unet_resnet34(encoder_weights=encoder_weights).to(device)
    teacher_model = build_unet_resnet34(encoder_weights=encoder_weights).to(device)

    copy_student_to_teacher(student_model, teacher_model)

    # Student-Encoder zunächst einfrieren
    set_encoder_trainable(student_model, trainable=False)

    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_model.eval()

    # -------------------------
    # Loss + Optimizer
    # -------------------------
    sup_criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=lr_phase1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        # -------------------------
    # Historie
    # -------------------------
    epoch_history = []
    train_total_loss_history = []
    train_sup_loss_history = []
    train_unsup_loss_history = []
    train_iou_history = []
    lambda_u_history = []

    val_student_loss_history = []
    val_student_iou_history = []
    val_teacher_loss_history = []
    val_teacher_iou_history = []

    best_val_iou_teacher = -1.0
    best_epoch_teacher = -1

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, epochs + 1):

        if epoch == freeze_epochs + 1:
            print(f"Unfreezing student encoder ab Epoche {epoch}")
            set_encoder_trainable(student_model, trainable=True)
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr_phase2)

        student_model.train()
        teacher_model.eval()

        train_total_loss = 0.0
        train_sup_loss = 0.0
        train_unsup_loss = 0.0
        train_iou = 0.0

        lambda_u = unsup_weight_max * sigmoid_rampup(epoch - 1, rampup_epochs)

        unlabeled_iter = cycle(unlabeled_loader)
        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for images_l, masks_l in pbar:
            unlabeled_batch = next(unlabeled_iter)

            images_l = images_l.to(device, non_blocking=True)
            masks_l = masks_l.to(device, non_blocking=True)

            images_u_weak = unlabeled_batch["image_weak"].to(device, non_blocking=True)
            images_u_strong = unlabeled_batch["image_strong"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits_l = student_model(images_l)
                sup_loss = sup_criterion(logits_l, masks_l)

                student_logits_u = student_model(images_u_strong)

                with torch.no_grad():
                    teacher_logits_u = teacher_model(images_u_weak)

                unsup_loss = consistency_loss_from_logits(
                    student_logits_u,
                    teacher_logits_u,
                    threshold=confidence_threshold,
                )

                total_loss = sup_loss + lambda_u * unsup_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_teacher(student_model, teacher_model, ema_decay=ema_decay)

            train_total_loss += total_loss.item()
            train_sup_loss += sup_loss.item()
            train_unsup_loss += unsup_loss.item()
            train_iou += iou_from_logits(logits_l.detach(), masks_l.detach())

            pbar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "sup": f"{sup_loss.item():.4f}",
                "unsup": f"{unsup_loss.item():.4f}",
                "lambda_u": f"{lambda_u:.3f}",
            })

        train_total_loss /= max(1, len(labeled_loader))
        train_sup_loss /= max(1, len(labeled_loader))
        train_unsup_loss /= max(1, len(labeled_loader))
        train_iou /= max(1, len(labeled_loader))

        val_loss_student, val_iou_student = validate(
            student_model, val_loader, sup_criterion, device
        )

        val_loss_teacher, val_iou_teacher = validate(
            teacher_model, val_loader, sup_criterion, device
        )

        epoch_history.append(epoch)
        train_total_loss_history.append(train_total_loss)
        train_sup_loss_history.append(train_sup_loss)
        train_unsup_loss_history.append(train_unsup_loss)
        train_iou_history.append(train_iou)
        lambda_u_history.append(lambda_u)

        val_student_loss_history.append(val_loss_student)
        val_student_iou_history.append(val_iou_student)
        val_teacher_loss_history.append(val_loss_teacher)
        val_teacher_iou_history.append(val_iou_teacher)

        print(
            f"\nEpoch {epoch}: "
            f"train_total={train_total_loss:.4f} | "
            f"train_sup={train_sup_loss:.4f} | "
            f"train_unsup={train_unsup_loss:.4f} | "
            f"train_iou={train_iou:.4f} | "
            f"lambda_u={lambda_u:.4f} || "
            f"val_student_loss={val_loss_student:.4f} | "
            f"val_student_iou={val_iou_student:.4f} || "
            f"val_teacher_loss={val_loss_teacher:.4f} | "
            f"val_teacher_iou={val_iou_teacher:.4f}"
        )

        if val_iou_teacher > best_val_iou_teacher:
            best_val_iou_teacher = val_iou_teacher
            best_epoch_teacher = epoch

            best_teacher_path = os.path.join(run_dir, "best_teacher.pth")
            torch.save({
                "epoch": epoch,
                "student_model_state": student_model.state_dict(),
                "teacher_model_state": teacher_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_iou_teacher": best_val_iou_teacher,
                "config": config,
            }, best_teacher_path)

            print(f"Bestes Teacher-Modell gespeichert: {best_teacher_path}")

    # -------------------------
    # Letzte Modelle speichern
    # -------------------------
    last_teacher_path = os.path.join(run_dir, "last_teacher.pth")
    torch.save({
        "epoch": epochs,
        "teacher_model_state": teacher_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_iou_teacher": best_val_iou_teacher,
        "config": config,
    }, last_teacher_path)

    last_student_path = os.path.join(run_dir, "last_student.pth")
    torch.save({
        "epoch": epochs,
        "student_model_state": student_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_iou_teacher": best_val_iou_teacher,
        "config": config,
    }, last_student_path)

    print(f"Letztes Teacher-Modell gespeichert: {last_teacher_path}")
    print(f"Letztes Student-Modell gespeichert: {last_student_path}")

    # -------------------------
    # CSV + Summary
    # -------------------------
    save_history_csv(
        run_dir,
        epoch_history,
        train_total_loss_history,
        train_sup_loss_history,
        train_unsup_loss_history,
        train_iou_history,
        lambda_u_history,
        val_student_loss_history,
        val_student_iou_history,
        val_teacher_loss_history,
        val_teacher_iou_history,
    )

    save_summary(
        run_dir=run_dir,
        best_epoch_teacher=best_epoch_teacher,
        best_val_iou_teacher=best_val_iou_teacher,
        final_train_total_loss=train_total_loss_history[-1],
        final_train_sup_loss=train_sup_loss_history[-1],
        final_train_unsup_loss=train_unsup_loss_history[-1],
        final_train_iou=train_iou_history[-1],
        final_val_student_loss=val_student_loss_history[-1],
        final_val_student_iou=val_student_iou_history[-1],
        final_val_teacher_loss=val_teacher_loss_history[-1],
        final_val_teacher_iou=val_teacher_iou_history[-1],
    )

    # -------------------------
    # Plots
    # -------------------------
    plot_metric(
        x=epoch_history,
        y_train=train_total_loss_history,
        y_val=val_teacher_loss_history,
        xlabel="Epoch",
        ylabel="Loss",
        title="Training Total Loss and Validation Teacher Loss over Epochs",
        train_label="Train Total Loss",
        val_label="Validation Teacher Loss",
        save_prefix=os.path.join(run_dir, "loss_over_epochs")
    )

    plot_metric(
        x=epoch_history,
        y_train=train_iou_history,
        y_val=val_student_iou_history,
        xlabel="Epoch",
        ylabel="IoU",
        title="Training IoU and Validation Student IoU over Epochs",
        train_label="Train IoU",
        val_label="Validation Student IoU",
        save_prefix=os.path.join(run_dir, "iou_student_over_epochs")
    )
    # Extra Plot: SSL-Loss-Komponenten
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_history, train_total_loss_history, label="Train Total Loss", linewidth=2)
    plt.plot(epoch_history, train_sup_loss_history, label="Train Supervised Loss", linewidth=2)
    plt.plot(epoch_history, train_unsup_loss_history, label="Train Unsupervised Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SSL Loss Components over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(run_dir, "ssl_loss_components_over_epochs.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(run_dir, "ssl_loss_components_over_epochs.eps"), format="eps", bbox_inches="tight")
    plt.close()

    print("Plot gespeichert als PNG:", os.path.join(run_dir, "ssl_loss_components_over_epochs.png"))
    print("Plot gespeichert als EPS:", os.path.join(run_dir, "ssl_loss_components_over_epochs.eps"))

    plot_metric(
        x=epoch_history,
        y_train=train_iou_history,
        y_val=val_teacher_iou_history,
        xlabel="Epoch",
        ylabel="IoU",
        title="Training IoU and Validation Teacher IoU over Epochs",
        train_label="Train IoU",
        val_label="Validation Teacher IoU",
        save_prefix=os.path.join(run_dir, "iou_teacher_over_epochs")
    )

    print("\nTraining abgeschlossen.")
    print(f"Best Teacher Val IoU: {best_val_iou_teacher:.4f} in Epoche {best_epoch_teacher}")
    print(f"Alle Ergebnisse gespeichert in: {run_dir}")


if __name__ == "__main__":
    main()