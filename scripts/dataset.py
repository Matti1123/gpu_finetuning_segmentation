import os
from PIL import Image
from torch.utils.data import Dataset
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

        self.images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.masks = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

        self.image_dict = {f.split(".")[0]: f for f in self.images}
        self.mask_dict = {f.split("_segmentation")[0]: f for f in self.masks}

        self.paired_keys = [k for k in self.image_dict if k in self.mask_dict]

        if len(self.paired_keys) == 0:
            print("Keine Bild-Maske Paare gefunden! Prüfe Ordner und dateinamen.")

        self.image_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paired_keys)

    def __getitem__(self, idx):
        key = self.paired_keys[idx]
        img_path = os.path.join(self.images_dir, self.image_dict[key])
        mask_path = os.path.join(self.masks_dir, self.mask_dict[key])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return image, mask