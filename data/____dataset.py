#### deprecated


from parser import *

import yaml
import numpy as np
import timeit
import tqdm
from torch.utils.data import DataLoader

ssh_9008_root='/usb/01.KITTI/semantic_kitti/'
pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
DATA = yaml.safe_load(open('../config/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('../config/config.yaml', 'r'))

train = 'train'
multi_dataset= SemanticKitti(root=ssh_9008_root,sequences=['3'],labels=DATA["labels"],
                            color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["multi"],max_points=ARCH["dataset"]["max_points"],train=train)

multi_dataloader = DataLoader(multi_dataset, batch_size=2, shuffle=False, num_workers=2,
                        pin_memory=True, drop_last=True)


original_dataset=SemanticKitti(root=ssh_9008_root,sequences=['3'],labels=DATA["labels"],
                            color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["single"],max_points=ARCH["dataset"]["max_points"],train=train)

original_dataloader = DataLoader(original_dataset, batch_size=2, shuffle=False, num_workers=2,
                        pin_memory=True, drop_last=True)




time_taken = timeit.default_timer()

overall_mean=0
overall_std=0
data_size = 0

feature_means = np.array([0, 0, 0, 0, 0])
feature_std = np.array([0, 0, 0, 0, 0])

multi_time_taken=[]
original_time_taken=[]

for batch_idx, (proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,
                scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v) in tqdm.tqdm(enumerate (multi_dataloader)):
    multi_time_taken.append(timeit.default_timer() - time_taken)
    time_taken = timeit.default_timer()


time_taken = timeit.default_timer()

for batch_idx, (proj_multi_temporal_scan,proj_multi_temporal_label,scan_points,scan_range,
                scan_remission,scan_labels,proj_single_label,pixel_u,pixel_v) in tqdm.tqdm(enumerate (original_dataloader)):

    original_time_taken.append(timeit.default_timer() - time_taken)
    time_taken = timeit.default_timer()

print(np.array(multi_time_taken).mean())
print(np.array(original_time_taken).mean())

overall_mean=feature_means/data_size
overall_std=feature_std/data_size

print("prog")