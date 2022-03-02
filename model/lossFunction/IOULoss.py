import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class IOULoss(nn.Module):
    def __init__(self, class_weight=None):
        super(IOULoss, self).__init__()
        self.class_weight = class_weight
        self.n_classes = len(class_weight)
        self.device = "cuda"
        self.reset_metric()

    def reset_metric(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def update_metric(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        # conf[self.ignore] = 0
        # conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def forward(self, x,y):
        """Computes the IOU loss.
           Args:
               y: a tensor of shape [B, 1, H, W] or [B,H,W] since we are using squeeze(1)
               logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
               # eps: added to the denominator for numerical stability.
           Returns:
               Loss = 1 - IOU score
           """
        y = torch.squeeze(y)
        x = torch.argmax(x, 1)

        # add-batch
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection / union).mean()

        iou = iou.clone().requires_grad_(True)
        iou_mean = iou_mean.clone().requires_grad_(True)

        loss =1



        return loss