import torch.nn.functional as F
import torch.nn as nn
import torch

## Pytorch_lightning ##
import pytorch_lightning as pl
import torchmetrics

## Load Network modules
from model.fusion import MRTnet_Fusion, LaserNet_Fusion, SalsaNext_Fusion

from model.modified_laser_net import LaserNet
from model.salsaNext import SalsaNext
from data.loss_w import weighted_loss
from model.lossFunction.DiceLoss import DiceLoss
from model.lossFunction.IOULoss import IOULoss
from data.iou import IouEval,AverageMeter





class LaserNet_pl(pl.LightningModule):
    def __init__(self, hparams):
        super(LaserNet_pl, self).__init__()

        print("Configuring model params")

        self.model= LaserNet(num_inputs=hparams.n_channels, channels = [64, 128, 128], num_outputs=hparams.n_classes)

        self.train_acc = torchmetrics.Accuracy()
        self.train_iou = torchmetrics.IoU(hparams.n_classes)
        self.train_f1 = torchmetrics.F1Score(num_classes=hparams.n_classes,mdmc_average='global',ignore_index=0)
        self.val_acc = torchmetrics.Accuracy()
        self.lossDice = DiceLoss()
        # self.ext_evaluator = iouEval(hparams.n_classes, "cuda", [])
        # self.xentropy = nn.CrossEntropyLoss(ignore_index=0)

        class_weight = torch.Tensor([0., 0.00369632, 0.13822922, 0.1152675, 0.05092939, 0.05742262,
                                     0.12050725, 0.14301027, 0.15536821, 0.00080676, 0.01025557, 0.00111224,
                                     0.03286304, 0.00120572, 0.00219725, 0.00060187, 0.02291232, 0.00203669,
                                     0.04180744, 0.09977031])
        self.w_xentropy = nn.CrossEntropyLoss(weight=class_weight,reduction='mean')

        # iou + miou (manual)
        self.acc = AverageMeter()
        self.miou = AverageMeter()
        self.iou = AverageMeter()
        self.evaluator = IouEval(n_classes=hparams.n_classes, device="cuda", ignore=0)

        self.class_names = ["unlabeled","car","bicycle","motorcycle","truck",
                            "other-vehicle","person","bicyclist","motorcyclist","road",
                            "parking","sidewalk","other-ground","building","fence",
                            "vegetation","trunk","terrain","pole","traffic-sign"]

        print("configuration finished")

    def forward(self, x):
        # input x shape: [BS,C,H,W] w/o Time and Ranges
        out = self.model(x)  ## class_pred_data.shape: [bs,C,H,W]
        return out

    # define custom loss function
    # loss cross_entropy
    def loss(self, y_hat, y):
        loss = self.w_xentropy(y_hat,y)
        # loss = self.xentropy(y_hat, y)
        return loss

    # focal loss
    def loss_fl(self, preds, labels, num_classes=20, size_average =False):

        # alpha is based on class proportion in dataset

        alpha = torch.Tensor([0., 22.931663513184, 857.562683105469, 715.110046386719, 315.961761474609, 356.245208740234,747.617004394531,
                                     887.223937988281,963.891540527344,5.005093097687,63.624687194824,6.900216579437,203.879608154297,7.480203628540,
                                     13.631549835205,3.733920812607,142.146163940430,12.635480880737,259.369873046875,618.966735839844])

        gamma=2
        assert  len(alpha) == num_classes
        alpha= torch.Tensor(alpha).cuda()

        # change shape to (x, n)
        preds = preds.permute(0,2,3,1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

    def loss_dice(self,yhat,y):
        return self.lossDice(y,yhat)


    def loss_IOU(self,logits,y):
        class_weight = torch.Tensor([0., 22.931663513184, 857.562683105469, 715.110046386719, 315.961761474609, 356.245208740234,747.617004394531,
                                     887.223937988281,963.891540527344,5.005093097687,63.624687194824,6.900216579437,203.879608154297,7.480203628540,
                                     13.631549835205,3.733920812607,142.146163940430,12.635480880737,259.369873046875,618.966735839844]).cuda()
        # iou_class = IOULoss(class_weight)
        # loss = iou_class(logits,y)

        preds = torch.argmax(logits, 1)
        self.evaluator.reset()
        self.evaluator.addBatch(preds, y)
        train_miou, train_iou = self.evaluator.getIoU()

        iou_inv = 1 - train_iou

        iou_loss = torch.dot(iou_inv.type(torch.float32),class_weight).sum()

        miou_inv = 1-train_miou

        return miou_inv.clone().requires_grad_(True)



    def training_step(self, batch, batch_idx):

        x, y = batch["proj_scan_only"], batch["proj_label_only"]
        y = y.long()
        # x = x.squeeze()
        logits = self(x) # this calls self.forward
        # loss = self.loss_IOU(logits,y)

        # loss=loss.grad_fn()
        loss = self.loss_fl(logits, y)
        # loss =  self.loss_dice(logits,y)
        # loss = self.loss_fl(logits,y)

        #log step metric
        # preds = torch.argmax(logits, 1)
        # acc=self.train_acc(preds, y)
        # iou=self.train_iou(preds, y)

        # f1=self.train_f1(preds,y)
        # self.log('train_acc_step', acc, on_step= True, on_epoch= False)
        # self.log('train_iou_step', iou, on_step=True, on_epoch=False)

        # log step metric
        preds = torch.argmax(logits, 1)
        self.evaluator.reset()
        self.evaluator.addBatch(preds, y)
        train_miou, train_iou = self.evaluator.getIoU()
        self.acc.update(self.evaluator.getacc())
        self.miou.update(train_miou)
        self.iou.update(train_iou)

        for i, iou in enumerate(self.iou.avg):
            self.log(' **{}**class**iou**'.format(self.class_names[i]), iou, on_step=True, on_epoch=False)

        self.log('train_step_accuracy', self.acc.val, on_step=True, on_epoch=False)
        self.log('train_step_miou', self.miou.val, on_step=True, on_epoch=False)

        self.log('train_loss', loss)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # for i, iou in enumerate(self.iou.avg):
        #     self.log('iou_{}'.format(self.class_names[i - 1]), iou, on_step=False, on_epoch=True)

        self.log('train_epoch_avg_accuracy', self.acc.avg, on_step=False, on_epoch=True)
        self.log('train_epoch_avg_miou', self.miou.avg, on_step=False, on_epoch=True)

        self.acc.reset()
        self.miou.reset()
        self.iou.reset()

        # sch = self.lr_schedulers()
        # # If the selected scheduler is a ReduceLROnPlateau scheduler.
        # if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     sch.step(self.miou.avg)

        #
        # self.log('train_epoch_accuracy', self.train_acc)
        # self.log('train_epoch_iou', self.train_iou)

    def test_step(self, batch, batch_idx):
        xs,y = batch["proj_scan_only"] , batch["proj_label_only"]
        logits = self(xs)

        # log step metric
        preds = torch.argmax(logits, 1)

        self.evaluator.reset()
        self.evaluator.addBatch(preds, y)
        self.acc.update(self.evaluator.getacc())
        train_miou, train_iou = self.evaluator.getIoU()
        self.miou.update(train_miou)
        self.iou.update(train_iou)

        self.log('test_step_accuracy', self.acc.val, on_step=True, on_epoch=False)
        self.log('test_step_miou', self.miou.val, on_step=True, on_epoch=False)


        for i, iou in enumerate(self.iou.avg):
            self.log('iou_{}'.format(self.class_names[i-1]), iou, on_step=True, on_epoch=False)

        return {'miou': self.miou.val}

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API

        self.log('test_epoch_avg_accuracy', self.acc.avg, on_step=False, on_epoch=True)
        self.log('test_epoch_avg_miou', self.miou.avg, on_step=False, on_epoch=True)

        self.acc.reset()
        self.miou.reset()
        self.iou.reset()

    def configure_optimizers(self):

        # optimizer = torch.optim.SGD(self.train_dicts,
        #                       lr=self.ARCH["train"]["lr"],
        #                       momentum=self.ARCH["train"]["momentum"],
        #                       weight_decay=self.ARCH["train"]["w_decay"])

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        return  [optimizer]

        #, "optimizer","lr_scheduler": scheduler, "monitor": monitor}



class SalsaNext_out(pl.LightningModule):
    def __init__(self, hparams):
        super(SalsaNext_out, self).__init__()
        # self.model=SalsaNext(num_channels=hparams.n_channels, num_classes=hparams.n_classes)
        self.model= SalsaNext_Fusion(hparams)
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1()
        self.train_recall = torchmetrics.Recall()
        self.train_precision = torchmetrics.Precision()
        self.train_iou = torchmetrics.IoU(hparams.n_classes)
        self.val_acc = torchmetrics.Accuracy()
        self.lossDice = DiceLoss()
        self.lr= hparams.lr

    def forward(self, x):
        # input x shape: [BS,C,H,W] w/o Time and Ranges
        out = self.model(x)  ## class_pred_data.shape: [bs,C,H,W]
        return out

    # define custom loss function
    # loss cross_entropy
    def loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y, ignore_index=-1)
        return loss

    # focal loss with weights
    def loss_fl(self, preds, labels, num_classes=20, size_average =False):

        # Loss weights list: weights for every classes which have been calculated in ./data/loss_w.py
        alpha = torch.Tensor([0., 0.00369632, 0.13822922, 0.1152675, 0.05092939, 0.05742262,
                              0.12050725, 0.14301027, 0.15536821, 0.00080676, 0.01025557, 0.00111224,
                              0.03286304, 0.00120572, 0.00219725, 0.00060187, 0.02291232, 0.00203669,
                              0.04180744, 0.09977031])

        gamma=2
        assert  len(alpha) == num_classes
        alpha= alpha.cuda()

        # change shape to (x, n)
        preds = preds.permute(0,2,3,1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

    def loss_dice(self,yhat,y):
        return self.lossDice(y,yhat)



    def training_step(self, batch, batch_idx):
        x, y = batch["proj_multi_temporal_scan"], batch["proj_multi_temporal_label"]
        y = y.long()
        x = x.squeeze()
        logits = self(x) # this calls self.forward
        loss = self.loss_dice(logits, y)

        #log step metric
        preds = torch.argmax(logits, 1)
        self.train_acc(preds, y)
        self.train_iou(preds, y)
        self.log('train_step_accuracy', self.train_acc, on_step= True, on_epoch= False)
        self.log('train_step_iou', self.train_iou, on_step=True, on_epoch=False)
        self.log('train_step_f1', self.train_f1, on_step=True, on_epoch=False)
        self.log('train_step_recall', self.train_recall, on_step=True, on_epoch=False)
        self.log('train_step_precision', self.train_precision, on_step=True, on_epoch=False)
        self.log('train_step_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_epoch_accuracy', self.train_acc)
        self.log('train_epoch_iou', self.train_iou)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer} #, "optimizer","lr_scheduler": scheduler, "monitor": monitor}





# visualize (copy it into testing function)
'''
# testing part
import yaml
from data.visualizer import visualizer
import numpy as np
import matplotlib.pyplot as plt

# only for testing
self.DATA = yaml.safe_load(open('config/semantic-kitti.yaml', 'r'))
self.visualize = visualizer(self.DATA["color_map"], "magma", self.DATA["learning_map_inv"])


xs, gt = batch["proj_scan_only"] , batch["proj_label_only"]
pred = self(xs)
pred_argmax = pred.argmax(dim=1)
pred_argmax = torch.squeeze(pred_argmax).cpu().numpy()
label_colormap = self.DATA["color_map"]
gt = self.visualize.map(gt, self.DATA["learning_map_inv"])
gt = np.squeeze(gt,0)
plt.figure(figsize=[20, 8], dpi=300)
plt.subplot(3, 1, 1)
plt.title("ground-truth")
ground_truth = np.array([[label_colormap[val] for val in row] for row in gt], dtype='B')
plt.imshow(ground_truth)
pred_argmax = self.visualize.map(pred_argmax, self.DATA["learning_map_inv"])
plt.subplot(3, 1, 2)
plt.title("predicted")
pred_argmax = np.array([[label_colormap[val] for val in row] for row in pred_argmax], dtype='B')
plt.imshow(pred_argmax)
plt.show()

'''

# in case of manual optimization
# self.automatic_optimization = False

# # manual optimization steps
#
# opt = self.optimizers()  # access the optimizer
# opt.zero_grad() # clear gradients from the previous training step
# loss = self.compute_loss(batch) # loss computation
# self.manual_backward(loss) # computes the d(loss)/d(param) for all params
# opt.step() #update the model params
