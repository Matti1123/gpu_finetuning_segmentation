import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Pfade anpassen ---
images_dir = "data/raw/images"
masks_dir = "data/raw/masks"
output_dir = "experiments"

os.makedirs(output_dir, exist_ok=True)

# Zielgröße
img_size = (256, 256)

# Anzahl Beispiele
max_images = 1

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

print(f"\n{'='*50}")
print("Bildgrößen (Original):")
print(f"{'='*50}\n")

for img_name in image_files[:max_images]:
    img_path = os.path.join(images_dir, img_name)

    # Original Bild laden
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (width, height)
    original_pixels = original_size[0] * original_size[1]

    print(f"Bild: {img_name}")
    print(f"  Größe: {original_size[0]} x {original_size[1]} Pixel")
    print(f"  Gesamtpixel: {original_pixels:,}")
    
    # Maske finden (mit "_segmentation" suffix)
    img_base_name = os.path.splitext(img_name)[0]
    mask_name = f"{img_base_name}_segmentation.png"
    mask_path = os.path.join(masks_dir, mask_name)
    
    # Maske laden (falls vorhanden)
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask_found = True
        print(f"  Maske: {mask_name}")
    else:
        mask_found = False
        print(f"  Maske nicht gefunden!")
    print()

    # Resize
    img_resized = img.resize(img_size)
    if mask_found:
        mask_resized = mask.resize(img_size, Image.NEAREST)

    # Plot
    if mask_found:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"Original Bild {original_size[0]}x{original_size[1]}")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(mask, cmap="gray")
        axes[0, 1].set_title(f"Original Maske {original_size[0]}x{original_size[1]}")
        axes[0, 1].axis("off")
        
        axes[1, 0].imshow(img_resized)
        axes[1, 0].set_title("Resized Bild (256x256)")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(mask_resized, cmap="gray")
        axes[1, 1].set_title("Resized Maske (256x256)")
        axes[1, 1].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img)
        axes[0].set_title(f"Original Bild {original_size[0]}x{original_size[1]}")
        axes[0].axis("off")
        
        axes[1].imshow(img_resized)
        axes[1].set_title("Resized Bild (256x256)")
        axes[1].axis("off")

    plt.tight_layout()

    base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_resize.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Gespeichert: {save_path}")

print("Fertig!")