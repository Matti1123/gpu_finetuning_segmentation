import os
import random

import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

from models.U_net_resnet34 import build_unet_resnet34
from scripts.dataset import ISICDataset


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_unet_resnet34()
    model.load_state_dict(checkpoint["teacher_model_state"])
    model.to(device)
    model.eval()

    return model


def save_prediction(image, mask, pred, save_path):
    """
    Speichert ein Bild mit 3 Subplots:
    links: Original Bild
    mitte: GT Maske (schwarz-weiß)
    rechts: Prediction (schwarz-weiß)
    """
    image = to_pil_image(image)
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    plt.figure(figsize=(15, 5))

    # Original Bild
    plt.subplot(1, 3, 1)
    plt.title("Original Image", fontsize=12)
    plt.imshow(image)
    plt.axis("off")

    # GT Maske
    plt.subplot(1, 3, 2)
    plt.title("GT Mask", fontsize=12)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # Prediction
    plt.subplot(1, 3, 3)
    plt.title("Prediction", fontsize=12)
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Pfade
    # -------------------------
    checkpoint_path = "runs/mean_teacher_unet_20pct/best_teacher.pt"
    save_dir = "results/mean_teacher_test"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = ISICDataset(
        images_dir="data/raw/images",
        masks_dir="data/raw/masks_selected",
        img_size=(256, 256),
    )

    # -------------------------
    # Modell laden (TEACHER!)
    # -------------------------
    model = load_model(checkpoint_path, device)

    # -------------------------
    # 5 zufällige Bilder
    # -------------------------
    indices = random.sample(range(len(dataset)), 5)

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]

        image = image.to(device).unsqueeze(0)
        mask = mask

        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

        pred = pred.cpu()

        save_path = os.path.join(save_dir, f"sample_{i}.png")

        save_prediction(
            image.squeeze().cpu(),
            mask,
            pred,
            save_path
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()