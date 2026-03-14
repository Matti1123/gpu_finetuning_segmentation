import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from scripts.dataset import ISICDataset

# ----------------------------
# Einstellungen
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "runs/exp_first/best.pt"
NUM_IMAGES = 10  # wie viele Beispiele anzeigen

# ----------------------------
# Modell laden
# ----------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # wir laden eigene Gewichte
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Modell geladen. Best Val IoU:", checkpoint["best_val_iou"])

# ----------------------------
# Dataset
# ----------------------------
dataset = ISICDataset(
    "data/raw/images",
    "data/raw/masks_selected",
    img_size=(256, 256),
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ----------------------------
# Visualisierung
# ----------------------------
with torch.no_grad():
    for i, (image, mask) in enumerate(loader):

        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(image)
        pred = torch.sigmoid(logits)
        pred = (pred > 0.5).float()

        # CPU für plotting
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_np)
        axes[0].set_title("Originalbild")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"runs/exp_first/prediction_{i}.png")

        if i + 1 >= NUM_IMAGES:
            break