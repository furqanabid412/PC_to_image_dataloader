
from data.parser import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml


def load_yml_settings(data_path='config/semantic-kitti.yaml',arch_path='config/config.yaml'):
    DATA = yaml.safe_load(open(data_path, 'r'))
    ARCH = yaml.safe_load(open(arch_path, 'r'))
    return DATA,ARCH


DATA,ARCH=load_yml_settings()
pc_root='E:/Datasets/Kitti-dataset'
log_dir_mypc = 'E:/work/pointcloud/TestCodes/hd_mrt_github/log'


dataset = SemanticKitti(root=pc_root, sequences=DATA["split"]["train"], labels=DATA["labels"],color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"], multi_proj=ARCH["single"],class_content=DATA["content"],
                        max_points=ARCH["dataset"]["max_points"], train='train',sort_by='two_way_sort')

data_frame = dataset[100]


print("testing-finished")
