import torch
import numpy as np
import yaml


class weighted_loss():
    def __init__(self):
        self.setup()

    def setup(self):
        # change this dir - if you change the project folders
        DATA = yaml.safe_load(open('config/semantic-kitti.yaml', 'r'))
        self.n_classes = len(DATA["learning_map_inv"])
        self.learning_map = DATA["learning_map"]
        self.class_content = DATA["content"]
        self.learning_ignore= DATA["learning_ignore"]

    def map_to_xentropy(self,label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
          if isinstance(data, list):
            nel = len(data)
          else:
            nel = 1
          if key > maxkey:
            maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
          lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
          lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
          try:
            lut[key] = data
          except IndexError:
            print("Wrong key ", key)
        # do the mapping
        return lut[label]


    def get_weights(self,epsilon_w = 0.001):
        content = torch.zeros(self.n_classes, dtype=torch.float)

        for cl, freq in self.class_content.items():
            x_cl = self.map_to_xentropy(cl,self.learning_map)  # map actual class to xentropy class
            print("class : ",x_cl,"freq",freq)
            content[x_cl] += freq


        loss_w = 1 / (content + epsilon_w)  # get weights

        for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
            if self.learning_ignore[x_cl]:
                # don't weigh
                loss_w[x_cl] = 0


        # print("Loss weights from content: ", loss_w.data)
        return  loss_w

    def get_original_weights(self):

        content = torch.zeros(self.n_classes, dtype=torch.float)

        for cl, freq in self.class_content.items():
            x_cl = self.map_to_xentropy(cl, self.learning_map)
            content[x_cl] += freq

        return content


# wl=weighted_loss()
#
# w =wl.get_weights()
#
# w=w.numpy()
#
# # w=w*100
#
# w = 1-w
#
# print("class_finished")

            ########################################################################