# Fusion Test
# in paper from Audi there are two fusion methods: LSTM and ConvGRU

import torch.nn.functional as F
import torch.nn as nn
import torch

## Pytorch_lightning ##
import pytorch_lightning as pl

## Load Network modules
from model.aptnet import APnet_CA_3
from model.modified_laser_net import LaserNet
from model.salsaNext import SalsaNext


## Headers ##
class Classification(nn.Module):
    def __init__(self, in_clannels, out_class=20):
        super(Classification, self).__init__()
        self.conv1 = nn.Conv2d(in_clannels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, out_class, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


# class StateEstimation(nn.Module):
#     def __init__(self, motion_category_num=2):
#         super(StateEstimation, self).__init__()
#         self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(32)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         return x
#
# class MotionPrediction(nn.Module):
#     def __init__(self, seq_len):
#         super(MotionPrediction, self).__init__()
#         self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(32)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         return x


## Fusion Network #1  ##
# Using STPN T: t-2, t-1, t / S: 3 ranges
class MRTnet_Fusion(pl.LightningModule):
    def __init__(self, hparams):
        super(MRTnet_Fusion, self).__init__()
        self.n_classes = hparams.n_classes
        self.n_clannels = hparams.n_channels

        self.apnet_1 = APnet_CA_3(in_channels=self.n_clannels)
        self.apnet_2 = APnet_CA_3(in_channels=self.n_clannels)
        self.apnet_3 = APnet_CA_3(in_channels=self.n_clannels)
        # self.max_pool = nn.AdaptiveMaxPool3d((None,None,1))
        self.bn = nn.BatchNorm2d(32 * 3)
        self.classify = Classification(32 * 3, self.n_classes)

        # self.res_block = nn.Sequential(nn.Conv2d(in_channels=hparams.n_classes,out_channels=hparams.n_classes,kernel_size=3, padding=1),
        #                               nn.BatchNorm2d(hparams.n_classes),
        #                               nn.ReLU(inplace=True))
        #
        # self.fusion = nn.Sequential(nn.Conv2d(32*3, 20, kernel_size=1, padding=0),
        #                             nn.BatchNorm2d(20),
        #                             *[self.res_block for _ in range(4)])

    def forward(self, x):
        # input single shape: [BS,T,C,H,W] with Time
        # input multi shape: [BS,T,R,C,H,W] with time and ranges

        x = x.permute(2, 0, 1, 3, 4, 5).contiguous()
        # shape:[R,BS,T,C,H,W]

        # Backbone network
        x_1 = self.apnet_1(x[0, :])  # -> x_1.shape: [BS,C,H,W,1]
        x_2 = self.apnet_2(x[1, :])
        x_3 = self.apnet_3(x[2, :])

        map_cat = torch.cat((x_1, x_2, x_3), dim=1).cuda()
        map_cat = self.bn(map_cat)
        # fusion_out = self.fusion(map_cat)
        fusion_out = self.classify(map_cat)

        # # Cell Classification head for range image
        # class_pred_data_1 = self.classify_1(x_1) # -> class_pred_data.shape: [BS,C,H,W]
        # class_pred_data_2 = self.classify_1(x_2)
        # class_pred_data_3 = self.classify_1(x_3)
        #
        # # Fusion
        # fusion_out = class_pred_data_1, class_pred_data_2, class_pred_data_3
        return fusion_out


## Prediction Network#3 For Test ##
class LaserNet_Fusion(pl.LightningModule):
    def __init__(self, hparams):
        super(LaserNet_Fusion, self).__init__()
        self.lasernet_1 = LaserNet(num_inputs=hparams.n_channels, channels=[64, 128, 128],
                                   num_outputs=hparams.n_classes)
        self.lasernet_2 = LaserNet(num_inputs=hparams.n_channels, channels=[64, 128, 128],
                                   num_outputs=hparams.n_classes)
        self.lasernet_3 = LaserNet(num_inputs=hparams.n_channels, channels=[64, 128, 128],
                                   num_outputs=hparams.n_classes)

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=hparams.n_classes, out_channels=hparams.n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(hparams.n_classes),
            nn.ReLU(inplace=True))

        self.bn = nn.BatchNorm2d(32 * 3)
        self.fusion = nn.Sequential(nn.Conv2d(32*3, 20, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(20),
                                    *[self.res_block for _ in range(4)])


    def forward(self, x):
        # input single shape: [BS,T,C,H,W] with Time
        # input multi shape: [BS,T,R,C,H,W] with time and ranges

        x_1 = self.lasernet_1(x[:, 1, :])  ## class_pred_data.shape: [bs,C,H,W]
        x_2 = self.lasernet_2(x[:, 0, :])  ## class_pred_data.shape: [bs,C,H,W]
        x_3 = self.lasernet_3(x[:, 0, :])  ## class_pred_data.shape: [bs,C,H,W]

        map_cat = torch.cat((x_1, x_2, x_3), dim=1).cuda()
        map_cat = self.bn(map_cat)  # bs,2*cls,64,1024
        fusion_out = self.fusion(map_cat)
        return fusion_out  # bs,cls,64,1024


## Prediction Network#3 For Test ##
class SalsaNext_Fusion(pl.LightningModule):
    def __init__(self, hparams):
        super(SalsaNext_Fusion, self).__init__()
        self.n_classes = hparams.n_classes
        self.salsanext_1 = SalsaNext(num_channels=hparams.n_channels, num_classes=hparams.n_classes)
        self.salsanext_2 = SalsaNext(num_channels=hparams.n_channels, num_classes=hparams.n_classes)
        self.salsanext_3 = SalsaNext(num_channels=hparams.n_channels, num_classes=hparams.n_classes)

        self.bn = nn.BatchNorm2d(32 * 3)
        self.classify = Classification(32 * 3, self.n_classes)

        ## from salsanext paper
        # self.res_block = nn.Sequential(nn.Conv2d(in_channels=hparams.n_classes,
        #                                          out_channels=hparams.n_classes, kernel_size=3, padding=1),
        #                                nn.BatchNorm2d(hparams.n_classes),
        #                                nn.ReLU(inplace=True))
        # self.bn = nn.BatchNorm2d(32 * 3)
        # self.fusion = nn.Sequential(nn.Conv2d(32 * 3, 20, kernel_size=1, padding=0),
        #                             nn.BatchNorm2d(20),
        #                             *[self.res_block for _ in range(4)])

    def forward(self, x):
        # input single shape: [BS,T,C,H,W] with Time
        # input multi shape: [BS,T,R,C,H,W] with time and range

        x_1 = self.salsanext_1(x[:, 0, :])  ## class_pred_data.shape: (bs,C,H,W) -> (bs,32,H,W)
        x_2 = self.salsanext_2(x[:, 1, :])
        x_3 = self.salsanext_3(x[:, 2, :])

        map_cat = torch.cat((x_1, x_2, x_3), dim=1).cuda()
        map_cat = self.bn(map_cat)
        fusion_out = self.classify(map_cat) ## bs,cls,64,1024
        return fusion_out  # bs,cls,64,1024