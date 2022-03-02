import torch.nn.functional as F
import torch.nn as nn
import torch

## Pytorch_lightning ##
import pytorch_lightning as pl
import torchmetrics

## Load Network modules
from model.fusion import MRTnet_multi_Fusion, LaserNet_Fusion
from data.loss_w import weighted_loss


__all__ = ["MRTnet_single_out", "MRTnet_multi_out", "UNet_out", "LaserNet_out"]


## Prediction#1  ##
class MRTnet_multi_out(pl.LightningModule):
    def __init__(self, hparams):
        super(MRTnet_multi_out, self).__init__()
        self.model = MRTnet_multi_Fusion(hparams)

    def forward(self, x):
        out= self.model(x)
        return out

    # define custom loss function
    def loss(self, y_pred_1, y_pred_2, y_pred_3 , y):
        loss = F.cross_entropy(y_pred_1, y[:,0,:])+\
               F.cross_entropy(y_pred_2, y[:,1,:])+\
               F.cross_entropy(y_pred_3, y[:,2,:])
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["gt_multi_pixel"]
        y = y.long()
        y_pred_1, y_pred_2, y_pred_3 = self(x) # this calls self.forward
        loss = self.loss(y_pred_1, y_pred_2, y_pred_3, y)

        # preds = torch.argmax(logits, 1)
        # # logging metrics we calculated by hand
        # self.log('train/loss', loss, on_epoch=True)
        # # logging a pl.Metric
        # self.train_acc(preds, ys)
        # self.log('train/acc', self.train_acc, on_epoch=True)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch["data"], batch["gt_multi_pixel"]
        y = y.long()
        y_pred_1, y_pred_2, y_pred_3 = self(x) # this calls self.forward
        loss = self.loss(y_pred_1, y_pred_2, y_pred_3, y)

        # self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        # self.log('valid/acc_epoch', self.valid_acc)
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    # def test_step(self, batch, batch_idx):
    #     xs, ys = batch
    #     logits, loss = self.loss(xs, ys)
    #     preds = torch.argmax(logits, 1)
    #
    #     self.test_acc(preds, ys)
    #     self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
    #     self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)
    #
    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    #     model_filename = "model_final.onnx"
    #     self.to_onnx(model_filename, dummy_input, export_params=True)
    #     wandb.save(model_filename)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
                                                               verbose=True,
                                                               threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)
        monitor='avg_val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}

## Prediction Network#2 ##
class MRTnet_single_out(pl.LightningModule):
    def __init__(self, hparams):
        super(MRTnet_single_out, self).__init__()
        self.n_classes= hparams.n_classes
        self.point_feat_size = hparams.n_channels
        self.lr = hparams.lr

        # self.motion_pred = MotionPrediction(seq_len=out_seq_len)
        # self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn_1 = STPN_CA_lite_3(in_channel=self.point_feat_size)
        self.stpn_2 = STPN_CA_lite_3(in_channel=self.point_feat_size)
        self.stpn_3 = STPN_CA_lite_3(in_channel=self.point_feat_size)
        self.max_pool = nn.AdaptiveMaxPool3d((None,None,1))
        self.bn = nn.BatchNorm2d(32)
        self.classify = Classification(self.n_classes)


    def forward(self, x):
        #  input x shape: [BS,T,R,C,H,W]
        x=x.permute(2,0,1,3,4,5).contiguous()  # shape:[R,BS,T,C,H,W]

        # Backbone network
        x_1 = self.stpn_1(x[0, :]).unsqueeze(-1) # -> x_1.shape: [BS,C,H,W,1]
        x_2 = self.stpn_2(x[1, :]).unsqueeze(-1)
        x_3 = self.stpn_3(x[2, :]).unsqueeze(-1)

        # Fusion block
        x_data= torch.cat((x_1,x_2,x_3),-1) # -> # x_data.shape: [BS,C,H,W,3]
        x_data=self.max_pool(x_data).squeeze() # -> x_data.shape: [BS,C,H,W]
        x_data= F.relu(self.bn(x_data))

        # Cell Classification head for range image
        class_pred_data = self.classify(x_data)  ## class_pred_data.shape: [bs,C,H,W]

        return class_pred_data

    # define custom loss function
    def loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y, ignore_index=-1)  if self.n_classes >1 else \
            F.binary_cross_entropy_with_logits(y_hat,y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["gt_single_pixel"]

        y = y.long()
        y_pred = self(x) # this calls self.forward
        torch.set_deterministic(False)
        loss = self.loss(y_pred, y)

        # preds = torch.argmax(logits, 1)
        # # logging metrics we calculated by hand
        # self.log('train/loss', loss, on_epoch=True)
        # # logging a pl.Metric
        # self.train_acc(preds, ys)
        # self.log('train/acc', self.train_acc, on_epoch=True)
        return {'loss':loss}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch["data"], batch["gt_single_pixel"]
    #     y = y.long()
    #     y_pred = self(x)  # this calls self.forward
    #     loss = self.loss(y_pred, y)
    #
    #     # self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
    #     # self.log('valid/acc_epoch', self.valid_acc)
    #     return {'val_loss': loss}
    #
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     return {'avg_val_loss': avg_loss}

    # def test_step(self, batch, batch_idx):
    #     xs, ys = batch
    #     logits, loss = self.loss(xs, ys)
    #     preds = torch.argmax(logits, 1)
    #
    #     self.test_acc(preds, ys)
    #     self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
    #     self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)
    #
    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    #     model_filename = "model_final.onnx"
    #     self.to_onnx(model_filename, dummy_input, export_params=True)
    #     wandb.save(model_filename)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
        #                                                        verbose=True,
        #                                                        threshold=0.00001, threshold_mode='rel', cooldown=0,
        #                                                        min_lr=0,
        #                                                        eps=1e-08)
        # monitor='avg_val_loss'
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler, "monitor": monitor}

## Prediction Network#3 For Test ##
class UNet_out(pl.LightningModule):
    def __init__(self, hparams):
        super(UNet_out, self).__init__()
        self.lr = hparams.lr
        self.unet= UNet(n_channels=hparams.n_channels, n_classes=hparams.n_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.train_iou = torchmetrics.IoU(hparams.n_classes)
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # input x shape: [BS,C,H,W] w/o Time and Ranges
        class_pred_data = self.unet(x)  ## class_pred_data.shape: [bs,C,H,W]
        return class_pred_data

    # define custom loss function
    def loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y, ignore_index=-1)
        return loss


    def training_step(self, batch, batch_idx):
        x, y = batch["proj_multi_temporal_scan"], batch["proj_single_label"]
        y = y.long()
        x = x.squeeze()
        logits = self(x) # this calls self.forward
        loss = self.loss(logits, y)

        #log step metric
        # preds = torch.argmax(logits, 1)
        # # logging metrics we calculated by hand
        # self.log('train/loss', loss, on_epoch=True)

        # self.train_acc(preds, ys)
        # self.log('train/acc', self.train_acc, on_epoch=True)
        preds = torch.argmax(logits, 1)
        self.train_acc(preds, y)
        self.train_iou(preds, y)
        self.log('train_acc_step', self.train_acc, on_step= True, on_epoch= False)
        self.log('train_iou_step', self.train_iou, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_epoch_accuracy', self.train_acc)
        self.log('train_epoch_iou', self.train_iou)

        # self.precision.compute()
        # self.log('train_epoch_precision', mean_precision)
        # self.precision.reset()

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch["proj_multi_temporal_scan"], batch["proj_single_label"]
    #     y = y.long()
    #     x = x.squeeze()
    #     logits = self(x) # this calls self.forward
    #     loss = self.loss(logits, y)
    #
    #     preds = torch.argmax(logits, 1)
    #     self.val_acc(preds, y)
    #     self.log('val_acc_step', self.val_acc, on_step=True, on_epoch=True)
    #     self.log('val_loss', loss)
    #     return {'val_loss': loss}
    #
    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([i['val_loss'] for i in outputs]).mean()
    #     return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.5)
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler, "monitor": monitor}


## 2022-02-02
## Prediction #3 For Test ##


class LaserNet_out(pl.LightningModule):
    def __init__(self, hparams):
        super(LaserNet_out, self).__init__()
        self.model= LaserNet_Fusion(hparams)
        self.train_acc = torchmetrics.Accuracy()
        self.train_iou = torchmetrics.IoU(hparams.n_classes)
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # input x shape: [BS,C,H,W] w/o Time and Ranges
        out = self.model(x)  ## class_pred_data.shape: [bs,C,H,W]
        return out

    # define custom loss function
    # loss cross_entropy
    def loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y, ignore_index=-1)
        return loss

    # focal loss
    def loss_fl(self, preds, labels, num_classes=20, size_average =False):
        alpha=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,
               0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] # list: weights for every classes
        w_l=weighted_loss()
        alpha = w_l.get_original_weights()


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


    def training_step(self, batch, batch_idx):
        x, y = batch["proj_multi_temporal_scan"], batch["proj_single_label"]
        y = y.long()
        x = x.squeeze()
        logits = self(x) # this calls self.forward
        loss = self.loss_fl(logits, y)

        #log step metric
        preds = torch.argmax(logits, 1)
        self.train_acc(preds, y)
        self.train_iou(preds, y)
        self.log('train_acc_step', self.train_acc, on_step= True, on_epoch= False)
        self.log('train_iou_step', self.train_iou, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_epoch_accuracy', self.train_acc)
        self.log('train_epoch_iou', self.train_iou)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.5)
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler, "monitor": monitor}

