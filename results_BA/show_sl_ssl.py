# results_BA/plot_sl_ssl_random.py

import os
import random
import glob
from PIL import Image

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from models.U_net_resnet34 import build_unet_resnet34


# =========================
# Konfiguration
# =========================

SL_MODEL_PATH = "results/supervised_training/run_20260501_190729_lr1e3_to_2e4_freeze7/best_model.pth"
SSL_MODEL_PATH = "results/semi_supervised_training/run_20260501_211831_mean_teacher_split20_thr08/best_teacher.pth" 

TEST_IMAGE_DIR = "data/testing/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = "data/testing/ISIC2018_Task1_Test_GroundTruth"

OUTPUT_DIR = "results_BA/qualitative_results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "random_sl_ssl_comparison.png")

IMG_SIZE = 256
THRESHOLD = 0.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Hilfsfunktionen
# =========================

def load_model(model_path, device):
    model = build_unet_resnet34()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def find_mask_path(image_path, mask_dir):
    stem = os.path.splitext(os.path.basename(image_path))[0]

    possible_masks = [
        os.path.join(mask_dir, stem + ".png"),
        os.path.join(mask_dir, stem + "_segmentation.png"),
        os.path.join(mask_dir, stem + ".jpg"),
        os.path.join(mask_dir, stem + ".jpeg"),
    ]

    for path in possible_masks:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Keine Maske gefunden für: {image_path}")


def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs >= THRESHOLD).float()

    return pred.squeeze().cpu().numpy()


def prepare_image_and_mask(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

    mask_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])

    image_tensor = image_transform(image)
    mask_tensor = mask_transform(mask)
    mask_tensor = (mask_tensor > 0.5).float()

    image_np = image_tensor.permute(1, 2, 0).numpy()
    mask_np = mask_tensor.squeeze().numpy()

    return image_tensor, image_np, mask_np


def plot_four_panel(image_np, mask_np, pred_sl, pred_ssl, save_path, image_name):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = [
        "Originalbild",
        "Ground Truth Maske",
        "Prediction SL",
        "Prediction SSL"
    ]

    images = [
        image_np,
        mask_np,
        pred_sl,
        pred_ssl
    ]

    cmaps = [
        None,
        "gray",
        "gray",
        "gray"
    ]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gespeichert: {save_path}")
    print(f"Bild: {image_name}")


# =========================
# Main
# =========================

def main():
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_paths = sorted(
        glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpeg")) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png"))
    )

    if len(image_paths) == 0:
        raise FileNotFoundError(f"Keine Testbilder gefunden in: {TEST_IMAGE_DIR}")

    image_path = random.choice(image_paths)
    mask_path = find_mask_path(image_path, TEST_MASK_DIR)

    image_tensor, image_np, mask_np = prepare_image_and_mask(image_path, mask_path)

    sl_model = load_model(SL_MODEL_PATH, device)
    ssl_model = load_model(SSL_MODEL_PATH, device)

    pred_sl = predict_mask(sl_model, image_tensor, device)
    pred_ssl = predict_mask(ssl_model, image_tensor, device)

    plot_four_panel(
        image_np=image_np,
        mask_np=mask_np,
        pred_sl=pred_sl,
        pred_ssl=pred_ssl,
        save_path=OUTPUT_PATH,
        image_name=os.path.basename(image_path)
    )


if __name__ == "__main__":
    main()