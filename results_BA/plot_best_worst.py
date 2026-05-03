import os
import glob
import csv
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from models.U_net_resnet34 import build_unet_resnet34


# =========================
# Paths anpassen falls nötig
# =========================

MODEL_PATH = "results/supervised_training/run_20260501_190729_lr1e3_to_2e4_freeze7/best_model.pth"

TEST_IMAGE_DIR = "data/testing/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = "data/testing/ISIC2018_Task1_Test_GroundTruth"

OUTPUT_DIR = "results_BA/best_worst_testset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 256
THRESHOLD = 0.5
BATCH_SIZE = 1


# =========================
# Dataset
# =========================

class TestSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )

        self.mask_dir = mask_dir
        self.img_size = img_size

        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(image_path))[0]

        possible_masks = [
            os.path.join(self.mask_dir, stem + ".png"),
            os.path.join(self.mask_dir, stem + "_segmentation.png"),
            os.path.join(self.mask_dir, stem + ".jpg"),
        ]

        mask_path = None
        for p in possible_masks:
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path is None:
            raise FileNotFoundError(f"Keine Maske gefunden für: {image_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)

        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor, image_path


# =========================
# IoU
# =========================

def calculate_iou(pred, mask, eps=1e-7):
    pred = pred.bool()
    mask = mask.bool()

    intersection = (pred & mask).sum().float()
    union = (pred | mask).sum().float()

    return ((intersection + eps) / (union + eps)).item()


# =========================
# Plot speichern
# =========================

def save_case(image, mask, pred, iou, image_path, save_path, title):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Originalbild")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title(f"Prediction\nIoU = {iou:.4f}")
    axes[2].axis("off")

    fig.suptitle(f"{title}: {os.path.basename(image_path)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# Main
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TestSegmentationDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Testbilder gefunden: {len(dataset)}")

    model = build_unet_resnet34()
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    best = {"iou": -1}
    worst = {"iou": 2}

    all_scores = []

    with torch.no_grad():
        for image, mask, image_path in dataloader:
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            probs = torch.sigmoid(logits)
            pred = (probs >= THRESHOLD).float()

            iou = calculate_iou(pred[0], mask[0])

            path = image_path[0]
            all_scores.append([os.path.basename(path), iou])

            if iou > best["iou"]:
                best = {
                    "iou": iou,
                    "image": image[0].cpu(),
                    "mask": mask[0].cpu(),
                    "pred": pred[0].cpu(),
                    "path": path,
                }

            if iou < worst["iou"]:
                worst = {
                    "iou": iou,
                    "image": image[0].cpu(),
                    "mask": mask[0].cpu(),
                    "pred": pred[0].cpu(),
                    "path": path,
                }

    save_case(
        best["image"],
        best["mask"],
        best["pred"],
        best["iou"],
        best["path"],
        os.path.join(OUTPUT_DIR, "best_case.png"),
        "Best Case"
    )

    save_case(
        worst["image"],
        worst["mask"],
        worst["pred"],
        worst["iou"],
        worst["path"],
        os.path.join(OUTPUT_DIR, "worst_case.png"),
        "Worst Case"
    )

    csv_path = os.path.join(OUTPUT_DIR, "iou_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "iou"])
        writer.writerows(all_scores)

    print("Fertig.")
    print(f"Best Case:  {os.path.basename(best['path'])} | IoU = {best['iou']:.4f}")
    print(f"Worst Case: {os.path.basename(worst['path'])} | IoU = {worst['iou']:.4f}")
    print(f"Gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()