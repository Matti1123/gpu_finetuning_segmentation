# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(256, 256)):
        """
        Dataset für ISIC 2018 Task 1 Segmentation
        images_dir: Pfad zu den Bildern (.jpg)
        masks_dir: Pfad zu den Masken (.png)
        img_size: Zielgröße (Höhe, Breite) für Images + Masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        # --- nur echte Bilder/Masken ---
        self.images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.masks = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

        # Dictionaries nach Bild-ID
        self.image_dict = {f.split(".")[0]: f for f in self.images}
        self.mask_dict = {f.split("_segmentation")[0]: f for f in self.masks}

        # Nur Paare, bei denen Bild UND Maske existieren
        self.paired_keys = [k for k in self.image_dict if k in self.mask_dict]

        if len(self.paired_keys) == 0:
            print("⚠️ Keine Bild-Maske Paare gefunden! Prüfe Ordner und Dateinamen.")

        # Transform für Resize
        self.transform = transforms.Resize(self.img_size)

    def __len__(self):
        return len(self.paired_keys)

    def __getitem__(self, idx):
        key = self.paired_keys[idx]
        img_path = os.path.join(self.images_dir, self.image_dict[key])
        mask_path = os.path.join(self.masks_dir, self.mask_dict[key])

        # --- Bilder laden ---
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 1 Kanal

        # Resize auf img_size
        image = self.transform(image)
        mask = self.transform(mask)

        # PyTorch Tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()  # binär

        return image, mask