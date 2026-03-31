import os
import math
import random
from itertools import cycle

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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

        logits = model(images)
        loss = criterion(logits, masks)

        val_loss += loss.item()
        val_iou += iou_from_logits(logits, masks)

    val_loss /= max(1, len(val_loader))
    val_iou /= max(1, len(val_loader))
    return val_loss, val_iou


def build_splits(full_dataset, val_fraction=0.2, label_fraction=0.2, seed=42):
    """
    full_dataset ist ISICDataset.
    Wir splitten über die paired_keys:
    - val
    - train
    - train_labeled
    - train_unlabeled
    """
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
    num_workers = 2

    epochs = 15
    lr = 1e-4
    ema_decay = 0.99

    unsup_weight_max = 0.1
    rampup_epochs = 12
    confidence_threshold = 0.8

    out_dir = "runs/mean_teacher_unet_20pct"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Volles gelabeltes Dataset laden
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

    # -------------------------
    # Labeled + Val via Subset
    # -------------------------
    labeled_dataset = Subset(full_dataset, labeled_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # -------------------------
    # Unlabeled Dataset:
    # Bildnamen aus full_dataset übernehmen
    # -------------------------
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
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Student + Teacher

    student_model = build_unet_resnet34().to(device)
    teacher_model = build_unet_resnet34().to(device)

    copy_student_to_teacher(student_model, teacher_model)

    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_model.eval()

    # -------------------------
    # Loss + Optimizer
    # -------------------------
    sup_criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_iou_teacher = -1.0

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, epochs + 1):
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
                # supervised branch
                logits_l = student_model(images_l)
                sup_loss = sup_criterion(logits_l, masks_l)

                # unsupervised branch
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

        # Validation: Student
        val_loss_student, val_iou_student = validate(
            student_model, val_loader, sup_criterion, device
        )

        # Validation: Teacher
        val_loss_teacher, val_iou_teacher = validate(
            teacher_model, val_loader, sup_criterion, device
        )

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

        # Bestes Teacher-Modell speichern
        if val_iou_teacher > best_val_iou_teacher:
            best_val_iou_teacher = val_iou_teacher
            ckpt_path = os.path.join(out_dir, "best_teacher.pt")

            torch.save({
                "epoch": epoch,
                "student_model_state": student_model.state_dict(),
                "teacher_model_state": teacher_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_iou_teacher": best_val_iou_teacher,
                "config": {
                    "img_size": img_size,
                    "val_fraction": val_fraction,
                    "label_fraction": label_fraction,
                    "batch_size_labeled": batch_size_labeled,
                    "batch_size_unlabeled": batch_size_unlabeled,
                    "epochs": epochs,
                    "lr": lr,
                    "ema_decay": ema_decay,
                    "unsup_weight_max": unsup_weight_max,
                    "rampup_epochs": rampup_epochs,
                    "confidence_threshold": confidence_threshold,
                    "seed": seed,
                }
            }, ckpt_path)

            print(f"✅ Best teacher saved: {ckpt_path} (val_iou={best_val_iou_teacher:.4f})")


if __name__ == "__main__":
    main()