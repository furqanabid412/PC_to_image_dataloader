import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, true, logits, eps=1e-7):
        """Computes the Sørensen–Dice loss.
           Note that PyTorch optimizers minimize a loss. In this
           case, we would like to maximize the dice loss so we
           return the negated dice loss.
           Args:
               true: a tensor of shape [B, 1, H, W]
               or [B,H,W] since we are using squeeze(1)-- my comments
               logits: a tensor of shape [B, C, H, W]. Corresponds to
                   the raw output or logits of the model.
               eps: added to the denominator for numerical stability.
           Returns:
               dice_loss: the Sørensen–Dice loss.
           """
        num_classes = logits.shape[1]

        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        # print(dice_loss)
        return (1 - dice_loss)