# Thanks MotionNet share their work.

import torch.nn.functional as F
import torch.nn as nn
import torch

__all__ = ["APnet_CBAM_3", "APnet_CA_3"]


# "STPN" is original from MotionNet.
# "STPN_Full" is modified network for any size of seq input (same to fig.3 in paper), efficiency decrease 40%

# "STPN_lite_3","STPN_lite_5" are designed for 3 seqs input and 5 seqs input, respectively.


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq, c, h, w)
        return x


class STPN_lite_5(nn.Module):
    def __init__(self, height_feat_size=13):  # height_feat_size
        super(STPN_lite_5, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))  # (batch*seq,64,h/2,w/2) -> (batch*seq,64,h/2,w/2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, seq-2, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * (seq-2), 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(
            self.bn2_1(self.conv2_1(x_1)))  # (batch * (seq-2), 64, h/2, w/2) -> (batch * (seq-2), 128, h/4, w/4)
        x_2 = F.relu(
            self.bn2_2(self.conv2_2(x_2)))  # (batch * (seq-2), 128, h/4, w/4) -> (batch * (seq-2), 128, h/4, w/4)

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, (seq-2), 128, h/4, w/4)
        x_2 = self.conv3d_2(x_2)  # (batch, (seq-2), 128, h/4, w/4) -> (batch, 1, 128, h/4, w/4)
        x_2 = x_2.squeeze()  # (batch, 128, h/4, w/4), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 256, h/8, w/8)
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  # (batch, 256, h/8, w/8) -> (batch, 256, h/8, w/8)

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))  # (batch, 256, h/8, w/8) -> (batch , 512, h/16, w/16)
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))  # (batch, 512, h/16, w/16) -> (batch , 512, h/16, w/16)

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch, 512+256, h/8, w/8) -> (batch, 256, h/8, w/8)
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch, 256, h/8, w/8)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch, 256, h/4, w/4)+ (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2),
                       x_1.size(3))  # (batch, (seq-2), 64, h/2, w/2) ->  (batch, seq-2 , 64, h/2, w/2)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 64, seq-2, h/2, w/2)
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))  # (batch, 64, 1, h/2, w/2)
        x_1 = x_1.squeeze()  # (batch, 64, h/2, w/2)
        # x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous() #  (batch, 1, 64, h/2, w/2)
        # x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous() #  (batch, 64, h/2, w/2)

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1),
                                                       dim=1))))  # (batch, 128 +64, h/2, w/2) -> (batch, 64, h/2, w/2)
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch, 64, h/2, w/2)

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch*seq, 32, h, w) -> (batch, seq, 32, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.squeeze()  # (batch, 32, h, w)
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, 32, h, w)
        # x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous() # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x),
                                                       dim=1))))  # (batch, 64 + 32, h, w) -> (batch, 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch, 32, h, w)

        return res_x  # (batch, 32, h, w)


class STPN_lite_3(nn.Module):  # seq number is 3
    def __init__(self, height_feat_size=13):
        super(STPN_lite_3, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))  # (batch*seq,64,h/2,w/2) -> (batch*seq,64,h/2,w/2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, (seq-2)=1, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * 1, 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))  # (batch, 64, h/2, w/2) -> (batch, 128, h/4, w/4)
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 256, h/8, w/8)
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  # (batch, 256, h/8, w/8) -> (batch, 256, h/8, w/8)

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))  # (batch, 256, h/8, w/8) -> (batch , 512, h/16, w/16)
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))  # (batch, 512, h/16, w/16) -> (batch , 512, h/16, w/16)

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch, 512+256, h/8, w/8) -> (batch, 256, h/8, w/8)
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch, 256, h/8, w/8)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch, 256, h/4, w/4)+ (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1),
                                                       dim=1))))  # (batch, 128 +64, h/2, w/2) -> (batch, 64, h/2, w/2)
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch, 64, h/2, w/2)

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch* seq, 32, h, w) -> (batch, seq, 32, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.squeeze()  # (batch, 32, h, w)
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, 32, h, w)
        # x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous() # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x),
                                                       dim=1))))  # (batch, 64 + 32, h, w) -> (batch, 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch, 32, h, w)

        return res_x  # (batch, 32, h, w)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

# candidate #02
# input : (batch, 3, 5, h, w)
# output : (batch, 32, h, w)

class APnet_CBAM_3(nn.Module):  # seq number is 3
    def __init__(self, num_channels =5, out_feature= 32):
        super(APnet_CBAM_3, self).__init__()
        self.conv_pre_1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # self.conv1_2 = CBAM(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = CBAM(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = CBAM(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = CBAM(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = CBAM(512)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = CBAM(256)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = CBAM(128)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = CBAM(64)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, out_feature, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        # self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        # self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        # self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        # self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        # self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        # self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        # self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = self.conv1_2(x_1)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, (seq-2)=1, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * 1, 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))  # (batch, 64, h/2, w/2) -> (batch, 128, h/4, w/4)
        # x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_2 = self.conv2_2(x_2)

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 256, h/8, w/8)
        # x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  # (batch, 256, h/8, w/8) -> (batch, 256, h/8, w/8)
        x_3 = self.conv3_2(x_3)

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))  # (batch, 256, h/8, w/8) -> (batch , 512, h/16, w/16)
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))  # (batch, 512, h/16, w/16) -> (batch , 512, h/16, w/16)
        x_4 = self.conv4_2(x_4)

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch, 512+256, h/8, w/8) -> (batch, 256, h/8, w/8)
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch, 256, h/8, w/8)
        x_5 = self.conv5_2(x_5)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch, 256, h/4, w/4)+ (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        # x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_6 = self.conv6_2(x_6)

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1),
                                                       dim=1))))  # (batch, 128 +64, h/2, w/2) -> (batch, 64, h/2, w/2)
        # x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch, 64, h/2, w/2)
        x_7 = self.conv7_2(x_7)

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch* seq, 32, h, w) -> (batch, seq, 32, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.squeeze()  # (batch, 32, h, w)
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, 32, h, w)
        # x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous() # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x),
                                                       dim=1))))  # (batch, 64 + 32, h, w) -> (batch, 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch, 32, h, w)

        return res_x  # (batch, 32, h, w)

# Coordinate Attention block

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        h, w = int(h), int(w)
        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_w = self.avg_pool_y(x)
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

# candidate #01
# input : (batch, 3, 5, h, w)
# output : (batch, 32, h, w)

class APnet_CA_3(nn.Module):  # seq number is 3
    def __init__(self, in_channels =5, out_feature= 32 ):
        super(APnet_CA_3, self).__init__()
        img_h, img_w = 64, 1024 # for Coordinate Attention
        self.conv_pre_1 = nn.Conv2d (in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # self.conv1_2 = CBAM(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = CA_Block(64, img_h / 2, img_w / 2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = CA_Block(128, img_h / 4, img_w / 4)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = CA_Block(256, img_h / 8, img_w / 8)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = CA_Block(512, img_h / 16, img_w / 16)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = CA_Block(256, img_h / 8, img_w / 8)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = CA_Block(128, img_h / 4, img_w / 4)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = CA_Block(64, img_h / 2, img_w / 2)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, out_feature, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        # self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        # self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        # self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        # self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        # self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        # self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        # self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(out_feature)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = self.conv1_2(x_1)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, (seq-2)=1, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * 1, 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))  # (batch, 64, h/2, w/2) -> (batch, 128, h/4, w/4)
        # x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_2 = self.conv2_2(x_2)

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  # (batch, 128, h/4, w/4) -> (batch, 256, h/8, w/8)
        # x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  # (batch, 256, h/8, w/8) -> (batch, 256, h/8, w/8)
        x_3 = self.conv3_2(x_3)

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))  # (batch, 256, h/8, w/8) -> (batch , 512, h/16, w/16)
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))  # (batch, 512, h/16, w/16) -> (batch , 512, h/16, w/16)
        x_4 = self.conv4_2(x_4)

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch, 512+256, h/8, w/8) -> (batch, 256, h/8, w/8)
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch, 256, h/8, w/8)
        x_5 = self.conv5_2(x_5)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch, 256, h/4, w/4)+ (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        # x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_6 = self.conv6_2(x_6)

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1),
                                                       dim=1))))  # (batch, 128 +64, h/2, w/2) -> (batch, 64, h/2, w/2)
        # x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch, 64, h/2, w/2)
        x_7 = self.conv7_2(x_7)

        x = x.reshape(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch* seq, 32, h, w) -> (batch, seq, 32, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.squeeze()  # (batch, 32, h, w)
        # x = x.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, 32, h, w)
        # x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous() # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x),
                                                       dim=1))))  # (batch, 64 + 32, h, w) -> (batch, 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch, 32, h, w)

        return res_x  # (batch, 32, h, w)


# For backup
####### Original implementation as belows ######
class STPN(nn.Module):
    def __init__(self, height_feat_size=13):  # height_feat_size
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))  # (batch*seq,64,h/2,w/2) -> (batch*seq,64,h/2,w/2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, seq-2, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * (seq-2), 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(
            self.bn2_1(self.conv2_1(x_1)))  # (batch * (seq-2), 64, h/2, w/2) -> (batch * (seq-2), 128, h/4, w/4)
        x_2 = F.relu(
            self.bn2_2(self.conv2_2(x_2)))  # (batch * (seq-2), 128, h/4, w/4) -> (batch * (seq-2), 128, h/4, w/4)

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, (seq-2), 128, h/4, w/4)
        x_2 = self.conv3d_2(x_2)  # (batch, (seq-2), 128, h/4, w/4) -> (batch, (seq-4), 128, h/4, w/4)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3),
                       x_2.size(4)).contiguous()  # (batch *  (seq-4), 128, h/4, w/4), seq = 1
        # because seq =5, (seq-4) =1. x_2 can be concatenated in decoder.
        # it is different in paper.

        # -- STC block 3
        x_3 = F.relu(
            self.bn3_1(self.conv3_1(x_2)))  # (batch * (seq-4), 128, h/4, w/4) -> (batch * (seq-4), 256, h/8, w/8)
        x_3 = F.relu(
            self.bn3_2(self.conv3_2(x_3)))  # (batch * (seq-4), 256, h/8, w/8) -> (batch * (seq-4), 256, h/8, w/8)

        # -- STC block 4
        x_4 = F.relu(
            self.bn4_1(self.conv4_1(x_3)))  # (batch * (seq-4), 256, h/8, w/8) -> (batch * (seq-4), 512, h/16, w/16)
        x_4 = F.relu(
            self.bn4_2(self.conv4_2(x_4)))  # (batch * (seq-4), 512, h/16, w/16) -> (batch * (seq-4), 512, h/16, w/16)

        # in paper, pooling is following after stc block 4. is it different ~?

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch * (seq-4), 512+256, h/8, w/8) -> (batch * (seq-4), 256, h/8, w/8)
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch * (seq-4), 256, h/8, w/8)

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))  # (batch, (seq-4), 128, h/4, w/4)
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 128, (seq-4), h/4, w/4)
        x_2 = F.adaptive_max_pool3d(x_2,
                                    (1, None, None))  # (batch, 128, (seq-4), h/4, w/4) -> (batch, 128, 1, h/4, w/4)
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 128, h/4, w/4)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch, 128, h/4, w/4)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch * (seq-4) , 256, seq, h/4, w/4)+ (batch, 128, h/4, w/4) ??? -> (batch * seq, 128, h/4, w/4)
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch * seq, 128, h/4, w/4) -> (batch * seq, 128, h/4, w/4)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2),
                       x_1.size(3))  # (batch * seq, 64, h, w) ->  (batch, seq, 64, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 64, seq, h, w)
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))  # (batch, 64, 1, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 64, h, w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch, 64, h, w)

        x_7 = F.relu(self.bn7_1(self.conv7_1(
            torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))  # (batch * seq, 128 +64, h/2, w/2)
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch * seq, 64, h/2, w/2)

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch*seq,32,h,w) -> (batch,seq,32,h,w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 32, h, w)
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()  # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(
            torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))  # ((batch * seq, 64 + 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch * seq, 32, h, w)

        return res_x  # (batch * seq, 32, h, w)


# For backup
# Reproduced STPN following paper
class STPN_mod(nn.Module):  # modified for any size of seq input (same to fig.3 in paper), efficiency decrease 40%
    def __init__(self, height_feat_size=13):  # height_feat_size represents height
        super(STPN_mod, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv3d_3 = Conv3D(256, 256, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv3d_4 = Conv3D(512, 512, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # (batch*seq,z,h,w)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))  # (batch*seq,z,h,w) ->(batch*seq,32,h,w)
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))  # -> (batch*seq,32,h,w)

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))  # (batch*seq, 32,h,w) -> (batch*seq,64,h/2,w/2)
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))  # (batch*seq,64,h/2,w/2) -> (batch*seq,64,h/2,w/2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, 64, h/2, w/2)
        x_1 = self.conv3d_1(x_1)  # (batch, seq, 64, h/2, w/2) -> (batch, seq, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, 64, h/2, w/2)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))  # (batch * seq, 64, h/2, w/2) -> (batch * seq, 128, h/4, w/4)
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))  # (batch * seq, 128, h/4, w/4) -> (batch * seq, 128, h/4, w/4)

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, 128, h/4, w/4)
        x_2 = self.conv3d_2(x_2)  # (batch, seq, 128, h/4, w/4) -> (batch, seq, 128, h/4, w/4)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, 128, h/4, w/4)
        # because seq =5, (seq-4) =1. x_2 can be concatenated in decoder.
        # it is different in paper.

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  # (batch * seq, 128, h/4, w/4) -> (batch * seq, 256, h/8, w/8)
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))  # (batch * seq, 256, h/8, w/8) -> (batch * seq, 256, h/8, w/8)

        x_3 = x_3.view(batch, -1, x_3.size(1), x_3.size(2), x_3.size(3)).contiguous()  # (batch, seq, 256, h/8, w/8)
        x_3 = self.conv3d_3(x_3)  # (batch, seq, 256, h/8, w/8) -> (batch, seq, 256, h/8, w/8)
        x_3 = x_3.view(-1, x_3.size(2), x_3.size(3), x_3.size(4)).contiguous()  # (batch * seq, 256, h/8, w/8)

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))  # (batch * seq, 256, h/8, w/8) -> (batch * seq, 512, h/16, w/16)
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))  # (batch * seq, 512, h/16, w/16) -> (batch * seq, 512, h/16, w/16)

        x_4 = x_4.view(batch, -1, x_4.size(1), x_4.size(2), x_4.size(3)).contiguous()  # (batch, seq, 512, h/16, w/16)
        x_4 = self.conv3d_4(x_4)  # (batch, seq, 64, h/2, w/2) -> (batch, seq, 512, h/16, w/16)
        x_4 = x_4.view(-1, x_4.size(2), x_4.size(3), x_4.size(4)).contiguous()  # (batch * seq, 512, h/16, w/16)

        x_4 = x_4.view(batch, -1, x_4.size(1), x_4.size(2), x_4.size(3))  # (batch, seq, 512, h/16, w/16)
        x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 512, seq, h/16, w/16)
        x_4 = F.adaptive_max_pool3d(x_4,
                                    (1, None, None))  # (batch, 512, seq, h/16, w/16) -> (batch, 512, 1, h/16, w/16)
        x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 512, h/16, w/16)
        x_4 = x_4.view(-1, x_4.size(2), x_4.size(3), x_4.size(4)).contiguous()  # (batch, 512, h/16, w/16)

        # -------------------------------- Decoder Path --------------------------------

        x_3 = x_3.view(batch, -1, x_3.size(1), x_3.size(2), x_3.size(3))  # (batch, seq, 256, h/8, w/8)
        x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 256, seq, h/8, w/8)
        x_3 = F.adaptive_max_pool3d(x_3, (1, None, None))  # (batch, 256, seq, h/8, w/8) -> (batch, 256, 1, h/8, w/8)
        x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 256, h/8, w/8)
        x_3 = x_3.view(-1, x_3.size(2), x_3.size(3), x_3.size(4)).contiguous()  # (batch, 256, h/8, w/8)

        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3),
                                                       dim=1))))  # (batch, 512, h/8, w/8)+ (batch, 256, h/8, w/8) -> (batch, 256, h/8, w/8)
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))  # (batch, 256, h/8, w/8)

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))  # (batch, seq, 128, h/4, w/4)
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 128, seq, h/4, w/4)
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))  # (batch, 128, seq, h/4, w/4) -> (batch, 128, 1, h/4, w/4)
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 128, h/4, w/4)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch, 128, h/4, w/4)

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2),
                                                       dim=1))))  # (batch, 256, seq, h/4, w/4)+ (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))  # (batch, 128, h/4, w/4) -> (batch, 128, h/4, w/4)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2),
                       x_1.size(3))  # (batch * seq, 64, h/2, w/2) ->  (batch, seq, 64, h/2, w/2)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 64, seq, h/2, w/2)
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))  # (batch, 64, 1, h/2, w/2)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 64, h/2, w/2)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch, 64, h/2, w/2)

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1),
                                                       dim=1))))  # (batch, 128, h/2, w/2) + (batch, 64, h/2, w/2)-> (batch,64, h/2, w/2)
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))  # (batch, 64, h/2, w/2)

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))  # (batch*seq,32,h,w) -> (batch,seq,32,h,w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 32, seq, h, w)
        x = F.adaptive_max_pool3d(x, (1, None, None))  # (batch, 32, 1, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, 1, 32, h, w)
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()  # (batch, 32, h, w)

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x),
                                                       dim=1))))  # (batch, 64, h, w) + (batch, 32, h, w)-> (batch, 32, h, w)
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))  # (batch, 32, h, w)

        return res_x  # (batch, 32, h, w)