import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ISICUnlabeledDataset(Dataset):
    def __init__(self, images_dir, img_size=(256, 256), image_list=None):
        """
        Unlabeled Dataset für Mean Teacher.
        Gibt zwei Augmentierungen desselben Bildes zurück.

        Keine geometrischen Augmentierungen:
        image_weak und image_strong bleiben räumlich deckungsgleich.
        """
        self.images_dir = images_dir
        self.img_size = img_size

        if image_list is not None:
            self.images = sorted(image_list)
        else:
            self.images = sorted([
                f for f in os.listdir(images_dir)
                if f.endswith(".jpg")
            ])

        # Teacher input: schwache / neutrale Augmentierung
        self.weak_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        # Student input: stärkere photometrische Augmentierung
        # Keine Rotation, kein Flip, kein Crop, kein Affine
        self.strong_transform = transforms.Compose([
            transforms.Resize(self.img_size),

            transforms.ColorJitter(
                brightness=0.40,
                contrast=0.40,
                saturation=0.30,
                hue=0.05
            ),

            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 1.5)
            ),

            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        image_weak = self.weak_transform(image)
        image_strong = self.strong_transform(image)

        return {
            "image_weak": image_weak,
            "image_strong": image_strong,
            "image_name": image_name,
        }