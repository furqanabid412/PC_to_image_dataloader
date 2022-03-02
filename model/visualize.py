import os
from argparse import ArgumentParser
import yaml
from data.data_interface import KittiDataModule
# from model.model_interface import UNet_out,MRTnet_multi_out,MRTnet_single_out, LaserNet_out
from model.simplified_model_interface import LaserNet_pl,SalsaNext_out
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# import wandb
import torch
import matplotlib.pyplot as plt
from data.parser import *
from data.visualizer import visualizer
import yaml
# wandb.login()

pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
log_dir_mypc = 'E:/work/pointcloud/TestCodes/hd_mrt_github/log'

# command line arguments
parser = ArgumentParser(add_help=False)
parser.add_argument('--dataset_dir', default=pc_root)
parser.add_argument('--log_dir', default=log_dir_mypc)
parser.add_argument('--n_channels', type=int, default=5)
parser.add_argument('--n_classes', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=8)
hparams = parser.parse_args()


# loading the trained model
model = LaserNet_pl(hparams)

model.load_state_dict(torch.load('../100_epochs_single_frame.pth'))
# model.load_state_dict(torch.load('../50_epochs_single_frame_lasernet.pth'))
model.cuda()
# model.eval()

#loading the data


DATA = yaml.safe_load(open('../config/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('../config/config.yaml', 'r'))

visualize = visualizer(DATA["color_map"],"magma",DATA["learning_map_inv"])

dataset = SemanticKitti(root=pc_root, sequences=['0','1','2','3','4','5','6','7','9','10'], labels=DATA["labels"],
                          color_map=DATA["color_map"], learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                          sensor=ARCH["dataset"]["sensor"], multi_proj=ARCH["single"],class_content=DATA["content"],
                          max_points=ARCH["dataset"]["max_points"], train='train',sort_by='normal')


frame_0 = dataset[0]

input = frame_0["proj_scan_only"]
input = input.expand(1,5,64,1024).cuda()
gt = frame_0["proj_label_only"]
pred = model(input)
pred_argmax = pred.argmax(dim=1)
pred_argmax = torch.squeeze(pred_argmax).cpu()
label_colormap = DATA["color_map"]

gt = visualize.map(gt, DATA["learning_map_inv"])

plt.figure(figsize=[20, 8], dpi=300)
plt.subplot(3, 1, 1)
plt.title("ground-truth")
ground_truth= np.array([[label_colormap[val] for val in row] for row in gt], dtype='B')
plt.imshow(ground_truth)

pred_argmax = visualize.map(pred_argmax, DATA["learning_map_inv"])
plt.subplot(3, 1, 2)
plt.title("predicted")
pred_argmax= np.array([[label_colormap[val] for val in row] for row in pred_argmax], dtype='B')
plt.imshow(pred_argmax)

plt.show()




# visualizing the kernals
'''
kernels = model.model.extract_1a.blocks[0].conv1.weight.detach().cpu()
fig, axarr = plt.subplots(kernels.size(0))
for idx in range(kernels.size(0)):
    ker= kernels[idx].squeeze()
    axarr[idx].imshow(kernels[idx].squeeze())
fig.show()
'''
print("test")
