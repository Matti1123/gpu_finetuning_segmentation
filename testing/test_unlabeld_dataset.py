import os
from cv2.gapi import mask
from matplotlib.pylab import sample
from numpy.random import sample
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from scripts.unlabeled_dataset import ISICUnlabeledDataset



# Optional: Maske laden
def load_mask(mask_dir, image_name):
    base = image_name.split(".")[0]
    mask_name = base + "_segmentation.png"
    mask_path = os.path.join(mask_dir, mask_name)
    mask_transform = transforms.Resize(
    (256, 256),
    interpolation=transforms.InterpolationMode.NEAREST)

    if os.path.exists(mask_path):
        from PIL import Image
        mask = Image.open(mask_path).convert("L")
        mask = mask_transform(mask)
        return mask
    else:
        return None


# --------------------------------------------------
# Plot Funktion
# --------------------------------------------------
def show_sample(sample, mask=None):
    image_weak = sample["image_weak"]
    image_strong = sample["image_strong"]

    image_weak = to_pil_image(image_weak)
    image_strong = to_pil_image(image_strong)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Weak")
    plt.imshow(image_weak)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Strong")
    plt.imshow(image_strong)
    plt.axis("off")

    if mask is not None:
        plt.subplot(1, 3, 3)
        plt.title("Mask (debug)")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("testing/sample.png",)      


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    images_dir = "data/raw/images"
    masks_dir = "data/raw/masks_selected"  # optional

    dataset = ISICUnlabeledDataset(
        images_dir=images_dir,
        img_size=(256, 256),
    )

    print(f"Dataset size: {len(dataset)}")

    # zufälliges Beispiel
    idx = torch.randint(0, len(dataset), (1,)).item()

    sample = dataset[idx]
    image_name = sample["image_name"]

    print("Image:", image_name)

    # optional Maske laden
    mask = load_mask(masks_dir, image_name)

    show_sample(sample, mask)

    diff = torch.abs(sample["image_weak"] - sample["image_strong"]).mean()
    print("Difference:",diff.item())
    print(sample["image_weak"].shape)
    print(sample["image_strong"].shape)
    print(mask.size)


if __name__ == "__main__":
    main()