import matplotlib.pyplot as plt
import numpy as np
from data.parser import *
import yaml
import torch
import os
import timeit
import tqdm
from torch.utils.data import DataLoader
import open3d
import cv2 as cv

class visualizer():
    def __init__(self,label_colormap,image_colormap,learning_map_inv):
        self.label_colormap = label_colormap
        self.image_colormap = image_colormap
        self.learning_map_inv = learning_map_inv

    def label_image_2D(self, input, title, scale, show_plot=True, save_image=False, path='test.jpg'):
        # H,W=np.shape(input)
        plt.figure(figsize=[80,7])
        plt.title(title,fontsize=70)
        image = np.array([[self.label_colormap[val] for val in row] for row in input], dtype='B')
        plt.imshow(image)
        if show_plot:
            plt.show()
        if save_image:
            plt.savefig(path)

    # 2D visualization any image

    def range_image_2D(self, input, title, scale, colormap="magma"):
        cmap = plt.cm.get_cmap(colormap, 10)
        H, W = np.shape(input)
        plt.figure(figsize=(int(W / scale) - 4, int(H / scale) + 2))
        plt.title(title)
        plt.imshow(input, cmap=cmap)
        plt.colorbar()
        plt.show()

    def pcl_3d(self,scan_points,scan_labels):

        pcd = open3d.geometry.PointCloud()
        scan_points = scan_points.numpy()
        scan_points = scan_points[scan_labels != -1, :]
        pcd.points = open3d.utility.Vector3dVector(scan_points)
        scan_labels = scan_labels.numpy()
        scan_labels = scan_labels[scan_labels != -1]
        scan_labels = self.map(scan_labels, self.learning_map_inv)
        colors = np.array([self.label_colormap[x] for x in scan_labels])

        pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        # vis.create_window(width=width, height=height, left=100)
        # vis.add_geometry(pcd)
        vis = open3d.visualization.draw_geometries([pcd])
        open3d.visualization.ViewControl()
    @staticmethod
    def map(label, mapdict):
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

def func_vis():
    # root='/home/share/dataset/semanticKITTI'
    pc_root='E:/Datasets/SemanticKitti/dataset/Kitti'
    # laptop_root ='/media/furqan/Terabyte/Lab/datasets/semanticKitti'
    DATA = yaml.safe_load(open('../config/semantic-kitti.yaml', 'r'))
    ARCH = yaml.safe_load(open('../config/config.yaml', 'r'))

    visualize = visualizer(DATA["color_map"],"magma",DATA["learning_map_inv"])

    dataset = SemanticKitti(root=pc_root, sequences=['0','1','2','3','4','5','6','7','9','10'], labels=DATA["labels"],
                              color_map=DATA["color_map"], learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                              sensor=ARCH["dataset"]["sensor"], multi_proj=ARCH["single"],class_content=DATA["content"],
                              max_points=ARCH["dataset"]["max_points"], train='train',sort_by='normal')

    # batch=dataset[4]

    # batch = dataset[4]

    label_colormap = DATA["color_map"]

    batch = dataset[4511]

    input = batch['proj_scan_only'][1].numpy()
    gt = batch['proj_label_only'].numpy()




    w,h=np.shape(gt)
    array1 = np.full((w, 256), 0, dtype=np.int32)
    array2 = np.full((w, 256), 0, dtype=np.int32)

    col1,col2 = 0,0
    for i in range(512):
        if i%2==0:
            array1[:,col1]=gt[:,i]
            col1+=1
        else:
            array2[:, col2] = gt[:, i]
            col2 += 1

    gt = visualize.map(gt, DATA["learning_map_inv"])
    plt.figure(figsize=[20,8],dpi=300)
    plt.subplot(3,1,1)
    plt.title("original")
    image = np.array([[label_colormap[val] for val in row] for row in gt], dtype='B')
    plt.imshow(image)

    gt1 = visualize.map(array1, DATA["learning_map_inv"])
    plt.subplot(3, 1, 2)
    plt.title("Even")
    image = np.array([[label_colormap[val] for val in row] for row in gt1], dtype='B')
    plt.imshow(image)

    gt2 = visualize.map(array2, DATA["learning_map_inv"])
    plt.subplot(3, 1, 3)
    plt.title("odd")
    image = np.array([[label_colormap[val] for val in row] for row in gt2], dtype='B')
    plt.imshow(image)




    plt.show()



    # cv.imwrite('image.jpg',image)
    # cv.imshow("image", image)
    # cv.waitKey(0)


    #
    # labels=batch['scan_labels'].numpy()
    #
    # for i in range(len(labels)):
    #     if labels[i]==8:
    #         print("xyz",batch['scan_points'][i])
    #
    # visualize.pcl_3d(batch['scan_points'],batch['scan_labels'])
    for i in range(1):
        batch = dataset[i+5]
        plt.figure(figsize=(40,20))
        for j in range(2):
            input = batch['proj_multi_range_only_scan'][j][1].numpy()
            plt.subplot(2,1,j+1)
            # gt = gt.numpy()
            plt.title("frame"+str(i)+','+str(j))
            gt = batch['proj_multi_range_only_label'][j].numpy()
            # gt = visualize.map(gt, DATA["learning_map_inv"])
            image = np.array([[label_colormap[val] for val in row] for row in gt], dtype='B')
            plt.imshow(image)
            # plt.colorbar()
            # visualize.label_image_2D(gt, "gt" + str(i), 10)
        plt.show()
    print("testing")


# func_vis()