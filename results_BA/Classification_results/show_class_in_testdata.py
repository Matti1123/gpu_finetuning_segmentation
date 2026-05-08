# testing/classify_segmentation_test_images.py

import os
import csv
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.U_net_resnet34 import build_unet_resnet34
from models.resnet_classifier import ResNet34ClassifierFromUNet


MODEL_PATH = "results/classifier_exp_7/best_classifier.pt"
UNET_CHECKPOINT_PATH = "runs/exp_first/best.pt"
TEST_IMAGE_DIR = "data/testing/ISIC2018_Task1-2_Test_Input"

OUTPUT_CSV = "results_BA/Classification_results/test_class_predictions_0.75.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.75
BATCH_SIZE = 16

CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]
class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, path


def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = TestImageDataset(TEST_IMAGE_DIR, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    unet = build_unet_resnet34()
    unet_checkpoint = torch.load(UNET_CHECKPOINT_PATH, map_location=DEVICE)

    if "model_state" in unet_checkpoint:
        unet.load_state_dict(unet_checkpoint["model_state"])
    else:
        unet.load_state_dict(unet_checkpoint)

    model = ResNet34ClassifierFromUNet(
        unet_model=unet,
        num_classes=7,
        freeze_encoder=False
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    rows = []
    counter_all = Counter()
    counter_above_threshold = Counter()

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Classifying test images"):
            images = images.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            max_probs, pred_indices = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                pred_idx = pred_indices[i].item()
                pred_class = CLASS_NAMES[pred_idx]
                max_prob = max_probs[i].item()

                counter_all[pred_class] += 1

                above_threshold = max_prob >= THRESHOLD

                if above_threshold:
                    counter_above_threshold[pred_class] += 1

                row = {
                    "image_path": paths[i],
                    "predicted_class": pred_class,
                    "probability": max_prob,
                    "above_0_75": above_threshold
                }

                for class_idx, class_name in enumerate(CLASS_NAMES):
                    row[f"prob_{class_name}"] = probs[i, class_idx].item()

                rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nAlle Testbilder nach vorhergesagter Klasse:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counter_all[class_name]}")

    print("\nNur Testbilder mit Softmax-Wahrscheinlichkeit >= 0.75:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counter_above_threshold[class_name]}")

    print(f"\nGesamtanzahl Testbilder: {len(dataset)}")
    print(f"Bilder >= 0.75: {sum(counter_above_threshold.values())}")
    print(f"CSV gespeichert unter: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()