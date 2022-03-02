import numpy as np
import os
import torch
from torch.utils.data import Dataset
from data.laserscan import *
import matplotlib.pyplot as plt


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

class SemanticKitti(Dataset):

  def __init__(self,
               root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               multi_proj,        # multi projection parameters
               class_content,    # array of proportion of specific class in point cloud
               max_points=150000,   # max number of points present in dataset
               train='train',
               sort_by='normal',
               has_image=True,
               ):
    # copying the params to self instance
    # self.root = os.path.join(root, "sequences")
    self.root = root
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]

    self.img_means = torch.tensor(sensor["img_means"])
    self.img_stds = torch.tensor(sensor["img_stds"])


    self.class_content = class_content
    self.sort_by = sort_by

    self.intervals = multi_proj["intervals"]
    self.timeframe = multi_proj["timeframes"]
    self.do_calib =multi_proj["calibrate"]
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.train = train

    self.has_image = has_image

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)

    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("No sequence folders exists in the directory "+self.root )

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []
    self.image_files = []

    # placeholder for calibration
    self.calibrations = []
    self.times = []
    self.poses = []
    self.frames_in_a_seq=[]
    frames_in_a_seq = []

    self.proj_matrix = []


    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      # print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      # sort for correspondance
      scan_files.sort()
      # append list
      self.scan_files.append(scan_files)

      # add image-files
      if self.has_image:
        image_path = os.path.join(self.root, seq, "image_2")
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if ".png" in f]
        self.image_files.append(image_files)

      # check all scans have labels
      if self.train == 'train':
        label_path = os.path.join(self.root, seq, "labels")
        label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_path)) for f in fn if is_label(f)]
        label_files.sort()
        # check all scans have labels
        assert (len(scan_files) == len(label_files))
        self.label_files.append(label_files)


      if self.do_calib :
        self.calibrations.append(self.parse_calibration(os.path.join(self.root, seq, "calib.txt")))  # read caliberation
        self.times.append(np.loadtxt(os.path.join(self.root, seq, 'times.txt'), dtype=np.float32))  # read times
        poses_f64 = self.parse_poses(os.path.join(self.root, seq, 'poses.txt'), self.calibrations[-1])
        self.poses.append([pose.astype(np.float32) for pose in poses_f64])  # read poses
        proj_matrix = np.matmul(self.calibrations[-1]["P2"], self.calibrations[-1]["Tr"])
        self.proj_matrix.append(proj_matrix)


      frames_in_a_seq.append(len(scan_files))


    self.frames_in_a_seq = np.array(frames_in_a_seq).cumsum()

    if self.intervals!=None:
      self.do_multi = True
    else:
      self.do_multi = False


    self.scan = LaserScan(interv=self.intervals,class_content=self.class_content, ch=5, H=self.sensor_img_H, W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down, tr=self.train,
                          is_proj=True,is_multi_proj=self.do_multi,sort_by=self.sort_by)  # default is True


  def parse_calibration(self, filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
      key, content = line.strip().split(":")
      values = [float(v) for v in content.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      calib[key] = pose
    calib_file.close()
    return calib

  def parse_poses(self, filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
      values = [float(v) for v in line.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses

  def get_seq_and_frame(self, index):
    # function takes index and convert it to seq and frame number

    if index < self.frames_in_a_seq[0]:
      return 0, index

    else:
      seq_count = len(self.frames_in_a_seq)
      for i in range(seq_count):
        fr = index + 1
        if i < seq_count - 1 and self.frames_in_a_seq[i] < fr and self.frames_in_a_seq[i + 1] > fr:
          # print("here")
          return i + 1, index - self.frames_in_a_seq[i]

        elif i < seq_count - 1 and self.frames_in_a_seq[i] == fr:
          return i, index - self.frames_in_a_seq[i - 1]

        elif i < seq_count - 1 and fr == self.frames_in_a_seq[-1]:
          return seq_count - 1, index - self.frames_in_a_seq[-2]


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

  def __len__(self):
    return self.frames_in_a_seq[-1]


  def get_pixel_features(self):

    # getting single projected scan
    single_proj_range = np.copy(self.scan.proj_range)
    single_proj_rem = np.copy(self.scan.proj_remission)
    single_proj_xyz = np.copy(self.scan.proj_xyz)
    proj_single_scan = np.concatenate((np.expand_dims(single_proj_rem, axis=0),np.expand_dims(single_proj_range, axis=0),
                                       np.rollaxis(single_proj_xyz, 2)),axis=0)
    proj_single_scan = torch.tensor(proj_single_scan)

    # getting mask of the projection
    single_proj_mask = np.copy(self.scan.proj_mask).astype(np.float)
    single_proj_mask = torch.tensor(single_proj_mask)

    proj_single_scan = (proj_single_scan - self.img_means[:, None, None]) / self.img_stds[:, None, None]
    proj_single_scan = proj_single_scan * single_proj_mask

    if self.train == 'train':
      proj_single_label = self.map(np.copy(self.scan.proj_sem_label), self.learning_map)
      proj_single_label = torch.tensor(proj_single_label)

    else:
      proj_single_label = torch.tensor([[]])

    return proj_single_scan,proj_single_label



  def get_point_features(self):
    # other params for post processing - only for frame "t"
    original_points = np.copy(self.scan.point)

    total_points = original_points.shape[0]

    scan_points = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    scan_points[:total_points] = torch.from_numpy(original_points)

    scan_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    scan_range[:total_points] = torch.from_numpy(np.copy(self.scan.unproj_range))

    scan_remission = torch.full([self.max_points], -1.0, dtype=torch.float)
    scan_remission[:total_points] = torch.from_numpy(np.copy(self.scan.remission))

    pixel_u = torch.full([self.max_points], -1, dtype=torch.long)
    pixel_u[:total_points] = torch.from_numpy(np.copy(self.scan.proj_x))

    pixel_v = torch.full([self.max_points], -1, dtype=torch.long)
    pixel_v[:total_points] = torch.from_numpy(np.copy(self.scan.proj_y))

    if self.train == 'train':
      # mapping classes and saving as a tensor
      original_labels = self.map(np.copy(self.scan.label), self.learning_map)

      scan_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      scan_labels[:total_points] = torch.from_numpy(np.squeeze(original_labels, axis=1))

    else:
      scan_labels = torch.tensor([])

    return scan_points,scan_range,scan_remission,pixel_u,pixel_v,scan_labels


  def __getitem__(self, index):

    seq, frame = self.get_seq_and_frame(index)

    # scan_paths=[]
    # label_paths=[]

    # get the list of filenames (scan and labels)
    # for multiple time frames

    pose0 = self.poses[seq][frame]


    # addded part (only for testing)
    self.scan.open_scan(self.scan_files[seq][frame], self.label_files[seq][frame], pose0, pose0, ego_motion=False)

    image = np.array(self.scan.loadImage(self.image_files[seq][frame]))

    pointcloud=np.hstack((self.scan.point,np.expand_dims(self.scan.remission,1)))
    sem_label = self.map(np.copy(self.scan.label), self.learning_map)
    sem_label =np.squeeze(sem_label, axis=1)

    proj_matrix=self.proj_matrix[seq]
    mapped_pointcloud, keep_mask = self.scan.mapLidar2Camera(proj_matrix,pointcloud[:, :3], image.shape[1], image.shape[0])

    y_data = mapped_pointcloud[:, 1].astype(np.int32)
    x_data = mapped_pointcloud[:, 0].astype(np.int32)

    combined = np.vstack((y_data,x_data))

    ind= np.unique(combined,axis=1,return_index=True,return_counts=True,return_inverse=False)

    counts=ind[2]
    counts =counts[counts>1]


    image = image.astype(np.float32) / 255.0
    # compute image view pointcloud feature
    depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
    keep_poincloud = pointcloud[keep_mask]
    proj_xyzi = np.zeros(
      (image.shape[0], image.shape[1], keep_poincloud.shape[1]), dtype=np.float32)
    proj_xyzi[x_data, y_data] = keep_poincloud
    proj_depth = np.zeros(
      (image.shape[0], image.shape[1]), dtype=np.float32)
    proj_depth[x_data, y_data] = depth[keep_mask]

    proj_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

    proj_label[x_data, y_data] = sem_label[keep_mask]

    # plt.imshow(proj_depth,cmap="magma")

    proj_label=self.map(proj_label,self.learning_map_inv)
    image = np.array([[self.color_map[val] for val in row] for row in proj_label], dtype='B')

    plt.imshow(image)
    plt.show()



    if self.timeframe > 1 :
      proj_multi_temporal_scan = []
      # comment 1 - for future use : we don;t need multi-temporal label(just we require single/multi range label) now
      # proj_multi_temporal_label = []


    for timeframe in range(self.timeframe):
      if frame - timeframe >= 0:
        curr_frame = frame - timeframe
      else:
        curr_frame = 0
      temp_scan_path = self.scan_files[seq][curr_frame]
      if self.train == 'train' :
        temp_label_path = self.label_files[seq][curr_frame]
      else:
        temp_label_path =[]

      curr_pose = self.poses[seq][curr_frame]

      # check whether a coordinate transformation is needed or not

      if timeframe == 0 or np.array_equal(pose0, curr_pose):
        ego_motion = False
      else:
        ego_motion = True

      # opening the scan(and label) file
      self.scan.open_scan(temp_scan_path,temp_label_path, pose0, curr_pose, ego_motion=ego_motion)

      if self.timeframe > 1:
        if self.do_multi:
          multi_proj_scan = np.copy(self.scan.concated_proj_range)
          proj_multi_temporal_scan.append(multi_proj_scan)

          # comment 2-1 - for future use : we don;t need multi-temporal label(just we require single/multi range label) now
          '''
          if self.train == 'train':
            multi_proj_label = np.copy(self.scan.concated_proj_semlabel)
            proj_multi_temporal_label.append(multi_proj_label)
          '''

          if timeframe == 0:
            # other params for post processing - only for frame "t"
            # proj_single_scan, proj_single_label = self.get_pixel_features()
            scan_points, scan_range, scan_remission, pixel_u, pixel_v, scan_labels = self.get_point_features()
            if self.train == 'train':
              proj_label = np.copy(self.scan.concated_proj_semlabel)
              proj_label = torch.from_numpy(proj_label)

        else:
          proj_single_scan, proj_single_label = self.get_pixel_features()
          proj_multi_temporal_scan.append(np.copy(proj_single_scan))

          if timeframe == 0:
            scan_points, scan_range, scan_remission, pixel_u, pixel_v, scan_labels = self.get_point_features()
            if self.train == 'train':
              proj_label = np.copy(proj_single_label)
              proj_label = torch.from_numpy(proj_label)

          # comment 2-2 - for future use : we don;t need multi-temporal label(just we require single/multi range label) now
          '''  
          if self.train == 'train':
            proj_multi_temporal_label.append(np.copy(proj_single_label))
          '''
      else:
        if self.do_multi:
          multi_proj_scan = np.copy(self.scan.concated_proj_range)
          if self.train == 'train':
            multi_proj_label = np.copy(self.scan.concated_proj_semlabel)
          scan_points, scan_range, scan_remission, pixel_u, pixel_v, scan_labels = self.get_point_features()
        else:
          proj_single_scan, proj_single_label = self.get_pixel_features()
          scan_points, scan_range, scan_remission, pixel_u, pixel_v, scan_labels = self.get_point_features()



    if self.timeframe > 1:
      proj_multi_temporal_scan = torch.tensor(np.copy(proj_multi_temporal_scan))

      # comment 3 - for future use : we don;t need multi-temporal label(just we require single/multi range label) now
      '''
      if self.train =='train':
        proj_multi_temporal_label = self.map(np.copy(proj_multi_temporal_label),self.learning_map)
        proj_multi_temporal_label = torch.tensor(proj_multi_temporal_label)
      else:
        proj_multi_temporal_label = torch.tensor([])
      '''

      if self.train == 'train':
        return {"proj_multi_temporal_scan": proj_multi_temporal_scan,
                "proj_multi_temporal_label": proj_label}

      else :
        return {"proj_multi_temporal_scan": proj_multi_temporal_scan,
                "proj_multi_temporal_label": proj_label,
                'scan_points': scan_points,
                'scan_range': scan_range,
                'scan_remission': scan_remission,
                'scan_labels': scan_labels,
                'pixel_u': pixel_u,
                'pixel_v': pixel_v}

    else:
      if self.do_multi:
        if  self.train == 'train':
          return {"proj_multi_range_only_scan": torch.tensor(multi_proj_scan),
                  "proj_multi_range_only_label": torch.tensor(multi_proj_label)}
        else:
          return {"proj_multi_range_only_scan": multi_proj_scan,
                  "proj_multi_range_only_label": multi_proj_label,
                  'scan_points': scan_points,
                  'scan_range': scan_range,
                  'scan_remission': scan_remission,
                  'scan_labels': scan_labels,
                  'pixel_u': pixel_u,
                  'pixel_v': pixel_v}

      else:
        if self.train == 'train':
          return {"proj_scan_only": proj_single_scan,
                  "proj_label_only": proj_single_label,
                  'scan_points': scan_points,
                  'scan_range': scan_range,
                  'scan_remission': scan_remission,
                  'scan_labels': scan_labels,
                  'pixel_u': pixel_u,
                  'pixel_v': pixel_v}
        else:
          return {"proj_scan_only": proj_single_scan,
                  "proj_label_only": proj_single_label,
                  'scan_points': scan_points,
                  'scan_range': scan_range,
                  'scan_remission': scan_remission,
                  'scan_labels': scan_labels,
                  'pixel_u': pixel_u,
                  'pixel_v': pixel_v}




    # return {"data": concat_time_frames,  # pixel features - size: BxTxRx5xHxW
    #         "single_data": single_proj_range,  # pixel features -size: Bx5xHxW
    #         "gt_single_pixel": proj_single_label,  # single pixel label - size: BxHxW
    #         "gt_multi_pixel": proj_multi_label,  # single pixel label - size: Bx5xHxW
    #         # "unproj_range": unproj_range, # range information of points - size: Bx100,000x1
    #         # "p_x": p_x,"p_y": p_y, # point location in image - size: Bx100,000
    #         # "points":curr_points, # point features - size : size: Bx100,000x3
    #         # "proj_range":proj_range_total, # point->pixel - size: BxHxW
    #         # "groundtruth_points": curr_lab # original gt of points - size : Bx100,000x1
    #         }


