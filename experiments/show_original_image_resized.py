import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Pfade anpassen ---
images_dir = "data/raw/images"
output_dir = "experiments"

os.makedirs(output_dir, exist_ok=True)

# Zielgröße
img_size = (256, 256)

# Anzahl Beispiele
max_images = 1

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

for img_name in image_files[:max_images]:
    img_path = os.path.join(images_dir, img_name)

    # Original laden
    img = Image.open(img_path).convert("RGB")

    # Resize
    img_resized = img.resize(img_size)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_resized)
    axes[1].set_title("Resized (256x256)")
    axes[1].axis("off")

    plt.tight_layout()

    base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_resize.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Gespeichert: {save_path}")

print("Fertig!")