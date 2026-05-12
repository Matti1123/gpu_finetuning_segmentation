import os
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# Pfade zu den Bildern
# =========================

IMAGE_1 = "data/raw/images/ISIC_0000097.jpg"
MASK_1 = "data/raw/masks/ISIC_0000097_segmentation.png"

IMAGE_2 = "data/raw/images/ISIC_0000093.jpg"
MASK_2 = "data/raw/masks/ISIC_0000093_segmentation.png"

OUTPUT_DIR = "results_BA/dataset_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Bilder laden
# =========================

img1 = Image.open(IMAGE_1).convert("RGB")
mask1 = Image.open(MASK_1).convert("L")

img2 = Image.open(IMAGE_2).convert("RGB")
mask2 = Image.open(MASK_2).convert("L")

# =========================
# Plot erstellen
# =========================

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
fig.patch.set_facecolor("white")

titles = [
    "Originalbild",
    "Ground Truth Maske",
    "Originalbild",
    "Ground Truth Maske"
]

images = [
    img1,
    mask1,
    img2,
    mask2
]

cmaps = [
    None,
    "gray",
    None,
    "gray"
]

for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=16)
    ax.axis("off")

plt.tight_layout(pad=1.0)

# =========================
# Speichern
# =========================

png_path = os.path.join(OUTPUT_DIR, "dataset_examples.png")
eps_path = os.path.join(OUTPUT_DIR, "dataset_examples.eps")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(eps_path, format="eps", dpi=300, bbox_inches="tight")

plt.close()

print("Fertig gespeichert:")
print(png_path)
print(eps_path)