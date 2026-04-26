# models/resnet34_classifier_from_unet.py

import torch
import torch.nn as nn


class ResNet34ClassifierFromUNet(nn.Module):
    def __init__(self, unet_model, num_classes=7, freeze_encoder=True):
        super().__init__()

        # Encoder aus deinem bereits trainierten U-Net übernehmen
        self.encoder = unet_model.encoder

        # Optional: Encoder einfrieren
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Klassifikationskopf
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes))

    def forward(self, x):
        # Encoder gibt mehrere Feature Maps zurück
        features = self.encoder(x)

        # letzte/tiefste Feature Map verwenden
        x = features[-1]          # z.B. [B, 512, 8, 8]

        # Global Average Pooling
        x = self.global_pool(x)   # [B, 512, 1, 1]

        # Flatten
        x = torch.flatten(x, 1)   # [B, 512]

        # Klassen-Logits
        x = self.classifier(x)    # [B, num_classes]

        return x