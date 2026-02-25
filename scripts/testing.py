# testing.py
import torch
from torch.utils.data import DataLoader
from dataset import ISICDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset mit Resize auf 256x256
dataset = ISICDataset(
    "data/raw/images",
    "data/raw/masks_selected",
    img_size=(256, 256)
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for i, (images, masks) in enumerate(loader):
    images = images.to(device)
    masks = masks.to(device)
    print(f"Batch {i}: images {images.shape}, masks {masks.shape}")
    if i == 2:  # nur 3 Batches testen
        break