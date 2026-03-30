import torch
import torch.nn as nn


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice Loss für binäre Segmentierung.
    logits:  [B, 1, H, W] - rohe Modell-Ausgabe
    targets: [B, 1, H, W] - binäre Maske (0 oder 1)
    """
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = dice_loss_from_logits(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss