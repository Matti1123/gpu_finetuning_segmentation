import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


# -------------------------
# Einstellungen
# -------------------------
images_dir = "data/raw/images"
image_name = "ISIC_0013023.jpg"  
output_dir = "experiments"
img_size = (256, 256)

os.makedirs(output_dir, exist_ok=True)


# -------------------------
# Genau wie im Dataloader
# -------------------------
weak_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

strong_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])


# -------------------------
# Bild laden
# -------------------------
image_path = os.path.join(images_dir, image_name)
image = Image.open(image_path).convert("RGB")

# Original resized für Vergleich
resized_pil = image.resize(img_size)

# Weak / Strong anwenden
image_weak = weak_transform(image)
image_strong = strong_transform(image)

# Tensor -> matplotlib Format
image_weak_np = image_weak.permute(1, 2, 0).numpy()
image_strong_np = image_strong.permute(1, 2, 0).numpy()


# -------------------------
# Plot erstellen
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(resized_pil)
axes[0].set_title("Resized")
axes[0].axis("off")

axes[1].imshow(image_weak_np)
axes[1].set_title("Weak Augmentation")
axes[1].axis("off")

axes[2].imshow(image_strong_np)
axes[2].set_title("Strong Augmentation")
axes[2].axis("off")

plt.tight_layout()

# Speichern
base_name = os.path.splitext(image_name)[0]
save_path = os.path.join(output_dir, f"{base_name}_weak_strong.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Gespeichert unter: {save_path}")