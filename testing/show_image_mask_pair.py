import os
import matplotlib.pyplot as plt
from PIL import Image

# --- Pfade anpassen ---
image_path = "/home/mci/gpu_finetuning_segmentation/data/raw/images/ISIC_0000017.jpg"
mask_path  = "/home/mci/gpu_finetuning_segmentation/data/raw/masks_selected/ISIC_0000017_segmentation.png"

save_dir = "results/plots"
os.makedirs(save_dir, exist_ok=True)

# --- Laden ---
image = Image.open(image_path).convert("RGB")
mask  = Image.open(mask_path).convert("L")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(image)
axes[0].set_title("Originalbild")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Ground Truth Maske")
axes[1].axis("off")

plt.tight_layout()

# --- Speichern ---
png_path = os.path.join(save_dir, "image_mask_pair.png")
eps_path = os.path.join(save_dir, "image_mask_pair.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", bbox_inches="tight")

plt.close()

print(f"Saved:\n{png_path}\n{eps_path}")