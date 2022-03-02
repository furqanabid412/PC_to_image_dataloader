from data.parser import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml

### Test Start ###


# DATA = yaml.safe_load(open('./config/semantic-kitti.yaml', 'r'))
# ARCH = yaml.safe_load(open('./config/config.yaml', 'r'))
### Test End ###

class KittiDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(KittiDataModule, self).__init__()
        self.batch_size = hparams.batch_size
        self.dataset_dir = hparams.dataset_dir
        self.num_workers= hparams.num_workers
        self.train_seq=['0','1','2','3','4','5','6','7','9','10']
        # self.train_seq = ['0']
        self.val_seq=['8']
        self.load_yml_settings()


    def load_yml_settings(self,data_path='./config/semantic-kitti.yaml',arch_path='./config/config.yaml'):
        self.DATA = yaml.safe_load(open(data_path, 'r'))
        self.ARCH = yaml.safe_load(open(arch_path, 'r'))


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.multi_dataset_train = SemanticKitti(root=self.dataset_dir, sequences=self.train_seq, labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"], learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"], multi_proj=self.ARCH["single"],class_content=self.DATA["content"],
                                          max_points=self.ARCH["dataset"]["max_points"], train='train',sort_by='two_way_sort')

            self.multi_dataset_val = SemanticKitti(root=self.dataset_dir, sequences=self.val_seq, labels=self.DATA["labels"],
                                                     color_map=self.DATA["color_map"], learning_map=self.DATA["learning_map"],
                                                     learning_map_inv=self.DATA["learning_map_inv"],
                                                     sensor=self.ARCH["dataset"]["sensor"], multi_proj=self.ARCH["single"],class_content=self.DATA["content"],
                                                     max_points=self.ARCH["dataset"]["max_points"], train='val',sort_by='two_way_sort')

        if stage == 'test':
            self.multi_dataset_test= SemanticKitti(root=self.dataset_dir, sequences=self.train_seq,
                                                     labels=self.DATA["labels"],
                                                     color_map=self.DATA["color_map"],
                                                     learning_map=self.DATA["learning_map"],
                                                     learning_map_inv=self.DATA["learning_map_inv"],
                                                     sensor=self.ARCH["dataset"]["sensor"],
                                                     multi_proj=self.ARCH["single"], class_content=self.DATA["content"],
                                                     max_points=self.ARCH["dataset"]["max_points"], train='train',
                                                     sort_by='two_way_sort')

    def train_dataloader(self):

        # print(len(self.multi_dataset_train))
        #
        # print(len(self.multi_dataset_val))

        multi_train_dataloader = DataLoader(self.multi_dataset_train,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            drop_last=True)
        return multi_train_dataloader

    def val_dataloader(self):
        multi_val_dataloader = DataLoader(self.multi_dataset_val,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=True)
        return multi_val_dataloader

    def test_dataloader(self):

        # for now only testing with training sequences

        multi_test_dataloader = DataLoader(self.multi_dataset_test,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            drop_last=True)
        return multi_test_dataloader