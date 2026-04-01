import torch

ckpt = torch.load("runs/deeplab_v3_plus/best.pt", map_location="cpu")
print("Beste Val IoU:", ckpt["best_val_iou"])
print("Epoche:", ckpt["epoch"])