import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from scripts.dataset import ISICDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "runs/exp_first/best.pt"

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

dataset = ISICDataset(
    "data/raw/images",
    "data/raw/masks_selected",
    img_size=(256, 256),
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,      # ggf. 2, 4, 8 testen
    pin_memory=True
)

def iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())

lowest_iou = 1.0
worst_image = None
worst_mask = None
worst_pred = None



with torch.inference_mode():
    for i, (image, mask) in tqdm(enumerate(loader), total=len(loader)):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(image)                 # [1,1,H,W]
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()          # binäre Maske

        iou = iou_from_logits(logits, mask)

        if iou < lowest_iou:
            lowest_iou = iou
            worst_image = image.detach().cpu().clone()
            worst_mask = mask.detach().cpu().clone()
            worst_pred = pred.detach().cpu().clone()

print("Schlechteste IoU:", lowest_iou)

img_np = worst_image[0].permute(1, 2, 0).numpy()
mask_np = worst_mask[0, 0].numpy()
pred_np = worst_pred[0, 0].numpy()


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(mask_np, cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
axes[2].imshow(pred_np, cmap="gray"); axes[2].set_title(f"Pred (IoU={lowest_iou:.3f})"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(f"results/worst_mask.png")

