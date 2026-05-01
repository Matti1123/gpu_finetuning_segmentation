import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

from models.U_net_resnet34 import build_unet_resnet34
from models.resnet_classifier import ResNet34ClassifierFromUNet
from scripts.classification_dataset import ISICClassificationDataset


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pfade
    image_dir = "data/testing/ISIC2018_Task3_Test_Input"
    csv_path = "data/testing/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
    model_path = "results/classifier_exp_6/best_classifier.pt"
    unet_checkpoint_path = "runs/exp_first/best.pt"

    batch_size = 16



    # KEINE Augmentation!
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ISICClassificationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # U-Net laden
    unet = build_unet_resnet34()
    checkpoint = torch.load(unet_checkpoint_path, map_location=device)

    if "model_state" in checkpoint:
        unet.load_state_dict(checkpoint["model_state"])
    else:
        unet.load_state_dict(checkpoint)

    # Klassifikationsmodell
    model = ResNet34ClassifierFromUNet(
        unet_model=unet,
        num_classes=7,
        freeze_encoder=False
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    bacc = balanced_accuracy_score(all_labels, all_preds)

    print(f"\nBalanced Accuracy: {bacc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()