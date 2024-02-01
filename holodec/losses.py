import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


# See also: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py

def load_loss(loss_name, split="training", reduction='mean'):

    supported = [
        "dice", "dice-bce", "iou", "focal", "tyversky",
        "focal-tyversky", "combo", "mse", "msle", "mae",
        "huber", "logcosh", "xtanh", "xsigmoid"
    ]

    logger.info(f"Loading {split} loss function {loss_name}")

    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "dice-bce":
        return DiceBCELoss()
    elif loss_name == "iou":
        return IoULoss()
    elif loss_name == "focal":
        return FocalLoss()
    elif loss_name == "tyversky":
        return TverskyLoss()
    elif loss_name == "focal-tyversky":
        return FocalTverskyLoss()
    elif loss_name == "combo":
        return ComboLoss()
    elif loss_name == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    elif loss_name == "msle":
        return MSLELoss(reduction=reduction)
    elif loss_name == "mae":
        return torch.nn.L1Loss(reduction=reduction)
    elif loss_name == "huber":
        return torch.nn.SmoothL1Loss(reduction=reduction)
    elif loss_name == "logcosh":
        return LogCoshLoss(reduction=reduction)
    elif loss_name == "xtanh":
        return XTanhLoss(reduction=reduction)
    elif loss_name == "xsigmoid":
        return XSigmoidLoss(reduction=reduction)
    else:
        raise OSError(
            f"Loss name {loss_name} not recognized. Please choose from {supported}")


class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        if weights is not None:
            weights = weights.reshape(-1)
            intersection = (inputs * targets * weights).sum()
        else:
            intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        if weights is not None:
            weights = weights.reshape(-1)
            intersection = (inputs * targets * weights).sum()
        else:
            intersection = (inputs * targets).sum()

        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        if weights is not None:
            weights = weights.reshape(-1)
            intersection = (inputs * targets * weights).sum()
        else:
            intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, weights=None, alpha=0.8, gamma=2, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Apply weights
        if weights is not None:
            weights = weights.reshape(-1)
            BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
            weighted_BCE = weights * BCE
        else:
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
            weighted_BCE = BCE

        BCE_EXP = torch.exp(-weighted_BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * weighted_BCE.mean()

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1, alpha=0.5, beta=0.5):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives & False Negatives
        # Apply weights
        if weights is not None:
            weights = weights.reshape(-1)
            TP = (inputs * targets * weights).sum()
            FP = ((1-targets) * inputs * weights).sum()
            FN = (targets * (1-inputs) * weights).sum()
        else:
            TP = (inputs * targets).sum()
            FP = ((1-targets) * inputs).sum()
            FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Apply weights
        if weights is not None:
            weights = weights.reshape(-1)
            weights /= weights.sum()
            TP = (inputs * targets * weights).sum()
            FP = ((1-targets) * inputs * weights).sum()
            FN = (targets * (1-inputs) * weights).sum()
        else:
            TP = (inputs * targets).sum()
            FP = ((1-targets) * inputs).sum()
            FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, weights=None, smooth=1, alpha=0.5, CE_RATIO=0.5, eps=1e-9):

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives & False Negatives
        if weights is not None:
            weights = weights.reshape(-1)
            intersection = (inputs * targets * weights).sum()
        else:
            intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (alpha * ((targets * torch.log(inputs)) +
                 ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12))) if self.reduction == 'mean' else torch.log(
            torch.cosh(ey_t + 1e-12))


class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(ey_t * torch.tanh(ey_t)) if self.reduction == 'mean' else ey_t * torch.tanh(ey_t)


class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t) if self.reduction == 'mean' else 2 * ey_t / (
                    1 + torch.exp(-ey_t)) - ey_t


class MSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSLELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        log_prediction = torch.log(prediction.abs() + 1)  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target, reduction=self.reduction)
        return loss
