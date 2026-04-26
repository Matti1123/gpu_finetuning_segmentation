import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset


class ISICClassificationDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.samples = []

        self.class_to_idx = {
            "MEL": 0,
            "NV": 1,
            "BCC": 2,
            "AKIEC": 3,
            "BKL": 4,
            "DF": 5,
            "VASC": 6,
        }

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_id = row["image"]

                label = None
                for class_name, idx in self.class_to_idx.items():
                    if float(row[class_name]) == 1.0:
                        label = idx
                        break

                if label is None:
                    continue  # safety

                image_path = os.path.join(image_dir, image_id + ".jpg")

                if os.path.exists(image_path):
                    self.samples.append((image_path, label))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)