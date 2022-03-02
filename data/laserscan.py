import numpy as np
from PIL import Image

# A class to load data from files and project it (multi_proj)
class LaserScan():
    def __init__(self, interv=[8,20],class_content=[0], ch=5, H=64, W=1024, fov_up=3.0, fov_down=-25.0, tr='train', is_proj=False, is_multi_proj=False,sort_by='normal'):

        self.is_project = is_proj
        self.is_multi_proj = is_multi_proj
        self.train = tr
        self.intervals = interv
        self.channels = ch  # need to change if we add more features
        self.class_content = class_content
        self.sort_by = sort_by

        self.proj_H = H
        self.proj_W = W
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.reset()

    def reset(self):
        self.complete_data = np.zeros((0, 4), dtype=np.float32)
        """ Reset scan members. """
        self.point = np.zeros((0, 3), dtype=np.float32)
        self.remission = np.zeros((0, 1), dtype=np.float32)
        self.label = np.zeros((0, 1), dtype=np.int32)
        self.label_ins = np.zeros((0, 1), dtype=np.int32)

        self.depth = np.zeros((0, 1), dtype=np.float32)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)  # projected range image - [H,W] range (-1 is no data)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)  # unprojected range (list of depths for each point)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)  # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)  # projected remission - [H,W] intensity (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)  # [H,W] index (-1 is no data)
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask
        self.proj_sem_label = np.full((self.proj_H, self.proj_W), 0, dtype=np.int32)
        self.proj_ins_label = np.full((self.proj_H, self.proj_W), 0, dtype=np.int32)

        # multi-ranges
        self.concated_proj_range = np.zeros((0, self.channels, self.proj_H, self.proj_W), dtype=np.float32)
        self.concated_proj_semlabel = np.zeros((0, self.proj_H, self.proj_W), dtype=np.int32)
        self.concated_proj_inslabel = np.zeros((0, self.proj_H, self.proj_W), dtype=np.int32)


    ########################################################################################################
    #                                 Data acquisition                                                     #
    ########################################################################################################

    def open_scan(self, pointcloud_path, label_path, pose0, curr_pose, ego_motion = False):
        self.reset()
        scan = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)

        if self.train == 'train':
            labels = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
            label = labels & 0xFFFF
            label_ins = labels >> 16

        else:
            label = np.zeros_like(scan[:, 1], dtype=int)
            label = np.expand_dims(label, 1)
            label_ins = np.zeros_like(scan[:, 1], dtype=int)
            label_ins = np.expand_dims(label_ins, 1)

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        if ego_motion:
            hpoints = np.hstack((points, np.ones_like(points[:, :1])))
            new_points = np.sum(np.expand_dims(hpoints, 2) * curr_pose.T, axis=1)
            new_points = new_points[:, :3]
            new_coords = new_points - pose0[:3, 3]
            new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
            points = np.hstack((new_coords, points[:, 3:]))
        self.set_points(points,remissions,label,label_ins)


    def set_points(self, point, remission, label, label_ins):
        self.point, self.remission, self.label, self.label_ins = point, remission, label, label_ins

        # if projection is wanted, then do it and fill in the structure
        if self.is_project:
            self.do_range_projection()


    def loadImage(self, path):
        return Image.open(path)

    def mapLidar2Camera(self, proj_matrx, pointcloud, img_h, img_w):
        # only keep point in front of the vehicle
        keep_mask = pointcloud[:, 0] > 0
        pointcloud_hcoord = np.concatenate([pointcloud[keep_mask], np.ones(
            [keep_mask.sum(), 1], dtype=np.float32)], axis=1)
        mapped_points = (proj_matrx @ pointcloud_hcoord.T).T  # n, 3
        # scale 2D points
        mapped_points = mapped_points[:, :2] / np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
        keep_idx_pts = (mapped_points[:, 0] > 0) * (mapped_points[:, 0] < img_h) * (mapped_points[:, 1] > 0) * (mapped_points[:, 1] < img_w)
        keep_mask[keep_mask] = keep_idx_pts
        # fliplr so that indexing is row, col and not col, row
        mapped_points = np.fliplr(mapped_points)

        return mapped_points[keep_idx_pts], keep_mask


    ########################################################################################################
    #                                 Range Projection                                                     #
    ########################################################################################################

    # function to convert points + labels into -> range image [complete]
    def do_range_projection(self):

        fov_up = self.fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        depth = np.linalg.norm(self.point, 2, axis=1)

        scan_x = self.point[:, 0]
        scan_y = self.point[:, 1]
        scan_z = self.point[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        # pitch = np.nan_to_num(pitch)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # store a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])

        if self.sort_by=='normal':
            order = np.argsort(depth)[::-1]

        else:
            # additional part for class(rarity) based sorting
            labelsq = np.squeeze(self.label, 1)
            class_c = np.zeros((len(self.label), 1), dtype=np.float32)
            for i in range(len(self.label)):
                curr = labelsq[i]
                class_c[i] = self.class_content[curr]
            order = np.lexsort((depth,np.squeeze(class_c,1)))[::-1]

        # class_content = self.class_content
        # label_w_content=class_content[self.label[:]].key()

        depth = depth[order]
        # copy of depth in descending order
        self.depth = np.copy(depth)
        indices = indices[order]
        points = self.point[order]
        remission = self.remission[order]
        sem_labels = self.label[order]
        ins_labels = self.label_ins[order]
        sem_labels = np.squeeze(sem_labels, axis=1)
        ins_labels = np.squeeze(ins_labels, axis=1)
        proj_y = proj_y[order]
        proj_x = proj_x[order]


        if self.is_multi_proj :
            # So the total range of depths will be "0" to "max"
            if self.intervals !=None :
                self.intervals_updated = [0] + self.intervals + [max(self.depth)]
            else :
                self.intervals_updated = [min(self.depth)]+[max(self.depth)]

            for index in range(len(self.intervals_updated)-1):
                lower,upper=self.intervals_updated[index],self.intervals_updated[index+1]

                rem = np.full((self.proj_H, self.proj_W),-1, dtype=np.float32)
                proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
                proj_xyz = np.full((self.proj_H, self.proj_W, 3),-1, dtype=np.float32)
                proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
                proj_ins_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

                mask = np.logical_and(depth > lower, depth <= upper)

                # 不能用projection以后的点来做吧？ 是否应该用projection之前的？

                proj_range[proj_y[mask],proj_x[mask]] = depth[mask]
                proj_xyz[proj_y[mask], proj_x[mask]] = points[mask]
                rem[proj_y[mask], proj_x[mask]] = remission[mask]

                if self.train == 'train':
                    proj_sem_label[proj_y[mask], proj_x[mask]] = sem_labels[mask]
                    proj_ins_label[proj_y[mask], proj_x[mask]] = ins_labels[mask]

                projection_cat = np.concatenate((np.expand_dims(rem, axis=0), np.expand_dims(proj_range, axis=0), np.rollaxis(proj_xyz, 2)), axis=0)
                self.concated_proj_range = np.concatenate((self.concated_proj_range, np.expand_dims(projection_cat, 0)), axis=0)
                self.concated_proj_semlabel = np.concatenate((self.concated_proj_semlabel, np.expand_dims(proj_sem_label, 0)), axis=0)
                self.concated_proj_inslabel = np.concatenate((self.concated_proj_inslabel, np.expand_dims(proj_ins_label, 0)), axis=0)

        # saving the single range data : this is from original rangenet++
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        self.proj_sem_label[proj_y, proj_x] = sem_labels
        self.proj_ins_label[proj_y, proj_x] = ins_labels