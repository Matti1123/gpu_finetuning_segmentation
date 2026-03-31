import torch
import torch.nn as nn


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice Loss für binäre Segmentierung.
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
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


def consistency_loss_from_logits(student_logits, teacher_logits, threshold=0.7, eps=1e-8):
    """
    Consistency Loss für Mean Teacher.
    Vergleicht Student- und Teacher-Vorhersagen auf ungelabelten Bildern.
    Es werden nur sichere Teacher-Pixel berücksichtigt.

    student_logits: [B, 1, H, W]
    teacher_logits: [B, 1, H, W]
    """
    student_probs = torch.sigmoid(student_logits)
    teacher_probs = torch.sigmoid(teacher_logits)

    # nur sichere Teacher-Pixel verwenden
    confidence_mask = ((teacher_probs > threshold) | (teacher_probs < (1.0 - threshold))).float()

    mse_map = (student_probs - teacher_probs) ** 2
    masked_mse = mse_map * confidence_mask

    denom = confidence_mask.sum() + eps
    return masked_mse.sum() / denom