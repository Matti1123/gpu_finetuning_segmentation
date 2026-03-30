import torch


@torch.no_grad()
def iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """
    IoU für binäre Segmentierung aus rohen Logits.
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


@torch.no_grad()
def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """
    Dice Score für binäre Segmentierung aus rohen Logits.
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (denominator + eps)
    return float(dice.mean().item())