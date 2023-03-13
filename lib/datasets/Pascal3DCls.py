import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip
from utils.img import Crop, DrawGaussian, Transform, Transform3D
import os
from imagecorruptions import corrupt
import random
from PIL import Image

# occ_level = 'FGL3_BGL3'
# occ_level = ''
using_occ_images_during_training = True
CLASS_NAMES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
class Pascal3D(data.Dataset):
  def __init__(self, opt, split, corruption=None):
    print('==> initializing pascal3d Star {} data.'.format(split))
    annot = {}
    tags = ['bbox', 'anchors', 'vis', 'dataset', 'class_id', 'imgname', 
            'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta', 'anchors_3d', 
            'space_embedding', 'truncated', 'occluded', 'difficult','valid']
    # tags = ['bbox', 'class_id', 'imgname', 
    #         'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta']
    self.corruption = corruption
    print("Using corruption:", self.corruption)
    if len(opt.occ) == 0:
      self.occ_level = ''
    else:
      self.occ_level = 'FGL%s_BGL%s' % (opt.occ, opt.occ)
    if len(self.occ_level) > 0:
        f = File('{}/Pascal3D/Pascal3D_{}-{}.h5'.format(ref.dataDir, self.occ_level, split), 'r')
    else:
        f = File('{}/Pascal3D/Pascal3D-{}.h5'.format(ref.dataDir, split), 'r')

    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']

    inds = []
    if split == 'train':
      inds = np.arange(len(annot['class_id']))
    else:
      inds = []
      for i in range(len(annot['class_id'])):
        # if annot['truncated'][i] < 0.5 and annot['occluded'][i] < 0.5 and annot['difficult'][i] < 0.5:
        if True:
          inds.append(i)

    for tag in tags:
      annot[tag] = annot[tag][inds]

    self.split = split
    self.opt = opt
    self.annot = annot
    self.nSamples = len(annot['class_id'])
    print('Loaded Pascal3D {} {} samples'.format(split, self.nSamples))
  
  def LoadImage(self, index):
    img_name = ''
    for v in range(len(self.annot['imgname'][index])):
      c = self.annot['imgname'][index][v]
      if c != 0:
        img_name += chr(c)
    occ_level = self.occ_level
    if len(occ_level) == 0 or self.split == 'train':
      path = '{}/Images/{}_{}/{}'.format(ref.pascal3dDir, ref.pascalClassName[self.annot['class_id'][index]], 
                                    ref.pascalDatasetName[self.annot['dataset'][index]], img_name)
    else:
      path = '/misc/lmbraid21/jesslen/StarMap/data/OccludedPASCAL3D/images/{}{}/{}'.format(ref.pascalClassName[self.annot['class_id'][index]], occ_level, img_name)
    # print(path)    
    img = cv2.imread(path)
    # print(path)
    return img
  
  
  def GetPartInfo(self, index):
    box = self.annot['bbox'][index].copy()
    c = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    s = max((box[2] - box[0]), (box[3] - box[1])) * ref.padScale
    v = np.array([self.annot['viewpoint_azimuth'][index], self.annot['viewpoint_elevation'][index], 
         self.annot['viewpoint_theta'][index]]) / 180.
    return c, s, v
      
  def __getitem__(self, index):
    img = self.LoadImage(index)
    class_id = self.annot['class_id'][index]
    # OOD
    # if class_id > 3:
      # class_id = class_id - 1
    c, s, v = self.GetPartInfo(index)
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0


    r = 0
    if self.split == 'train':
      s = s * (2 ** Rnd(ref.scale))
      c[1] = c[1] + Rnd(ref.shiftY)
      r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
      v[2] += r / 180.
      v[2] += 2 if v[2] < -1 else (-2 if v[2] > 1 else 0)
    inp = Crop(img, c, s, r, self.opt.inputRes)
    if self.corruption is not None:
        current_img = np.array(inp).astype(np.float32)
        # ensure range is 0-255 and convert to uint8
        current_img = np.array((current_img - current_img.min()) / (current_img.max() - current_img.min()) * 255, dtype=np.uint8)
        inp = corrupt(current_img, corruption_name=self.corruption, severity=4)
    
    inp = inp.transpose(2, 0, 1).astype(np.float32) / 255.

    if self.split == 'train':
      if np.random.random() < 0.5:
        inp = Flip(inp)
        v[0] = - v[0]
        v[2] = - v[2]
        v[2] += 2 if v[2] <= -1 else 0
    #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/master/rcnnVp/rcnnBinnedJointTrainValTestCreate.m#L77
    vv = v.copy()
    if vv[0] < 0:
      v[0] = self.opt.numBins - 1 - np.floor(-vv[0] * self.opt.numBins / 2.)
    else:
      v[0] = np.floor(vv[0] * self.opt.numBins / 2.)
    v[1] = np.ceil(vv[1] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v[2] = np.ceil(vv[2] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v = v.astype(np.int32)
    if self.opt.specificView:
      vv = np.ones(3 * len(ref.pascalClassId), dtype = np.int32) * self.opt.numBins
      vv[class_id * 3: class_id * 3 + 3] = v.copy()
      v = vv.copy()
    return inp, v, class_id


  def __len__(self):
    return self.nSamples

# occ_level = 'FGL3_BGL3'
# occ_level = ''
using_occ_images_during_training = True
CLASS_NAMES_OOD = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]
class Pascal3DOOD(data.Dataset):
  def __init__(self, opt, split, nuisance=None):
    print('==> initializing pascal3d Star {} data.'.format(split))
    annot = {}
    # tags = ['bbox', 'anchors', 'vis', 'dataset', 'class_id', 'imgname', 
    #         'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta', 'anchors_3d', 
    #         'space_embedding', 'truncated', 'occluded', 'difficult','valid']
    tags = ['bbox', 'class_id', 'imgname', 
            'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta']
    self.nuisance = nuisance
    print("Using disturbance:", self.nuisance)
    if len(opt.occ) == 0:
      self.occ_level = ''
    else:
      self.occ_level = self.nuisance
    if split == 'train':
      f = File('{}/OOD_train/Pascal3D-train.h5'.format(ref.dataDir), 'r')
    else:
      f = File('{}/OOD_test/Pascal3D_{}-val.h5'.format(ref.dataDir, self.nuisance), 'r')


    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']

    inds = []
    if split == 'train':
      inds = np.arange(len(annot['class_id']))
    else:
      inds = []
      for i in range(len(annot['class_id'])):
        # if annot['truncated'][i] < 0.5 and annot['occluded'][i] < 0.5 and annot['difficult'][i] < 0.5:
        if True:
          inds.append(i)

    for tag in tags:
      annot[tag] = annot[tag][inds]

    self.split = split
    self.opt = opt
    self.annot = annot
    self.nSamples = len(annot['class_id'])
    print('Loaded Pascal3D {} {} samples'.format(split, self.nSamples))
    print(set(self.annot['class_id']))
  
  def LoadImage(self, index):
    img_name = ''
    for v in range(len(self.annot['imgname'][index])):
      c = self.annot['imgname'][index][v]
      if c != 0:
        img_name += chr(c)
    if self.split == 'train':
      path = '/misc/lmbraid21/jesslen/Pascal3DforNeMo/dataOld/OOD_train_NeMo/images/{}/{}'.format(ref.pascalClassName[self.annot['class_id'][index]], img_name)
    else:
      path = '/misc/lmbraid21/jesslen/Pascal3DforNeMo/dataOld/OOD_test_NeMo/nuisances/{}/images/{}/{}'.format(self.nuisance,CLASS_NAMES_OOD[self.annot['class_id'][index]], img_name)



    if not os.path.exists(path):
      path = path.replace('JPEG', 'jpg')    
    img = cv2.imread(path)


    # print(path)
    return img
  
  
  def GetPartInfo(self, index):
    box = self.annot['bbox'][index].copy()
    c = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    s = max((box[2] - box[0]), (box[3] - box[1])) * ref.padScale
    v = np.array([self.annot['viewpoint_azimuth'][index], self.annot['viewpoint_elevation'][index], 
         self.annot['viewpoint_theta'][index]]) / 180.
    return c, s, v
      
  def __getitem__(self, index):
    img = self.LoadImage(index)
    class_id = self.annot['class_id'][index]
    if class_id > 3:
      class_id = class_id - 1
    c, s, v = self.GetPartInfo(index)
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0


    r = 0
    if self.split == 'train':
      s = s * (2 ** Rnd(ref.scale))
      c[1] = c[1] + Rnd(ref.shiftY)
      r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
      v[2] += r / 180.
      v[2] += 2 if v[2] < -1 else (-2 if v[2] > 1 else 0)
    inp = Crop(img, c, s, r, self.opt.inputRes)
    inp = inp.transpose(2, 0, 1).astype(np.float32) / 255.

    if self.split == 'train':
      if np.random.random() < 0.5:
        inp = Flip(inp)
        v[0] = - v[0]
        v[2] = - v[2]
        v[2] += 2 if v[2] <= -1 else 0
    #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/master/rcnnVp/rcnnBinnedJointTrainValTestCreate.m#L77
    vv = v.copy()
    if vv[0] < 0:
      v[0] = self.opt.numBins - 1 - np.floor(-vv[0] * self.opt.numBins / 2.)
    else:
      v[0] = np.floor(vv[0] * self.opt.numBins / 2.)
    v[1] = np.ceil(vv[1] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v[2] = np.ceil(vv[2] * self.opt.numBins / 2. + self.opt.numBins / 2. - 1)
    v = v.astype(np.int32)
    if self.opt.specificView:
      vv = np.ones(3 * len(ref.pascalClassId), dtype = np.int32) * self.opt.numBins
      vv[class_id * 3: class_id * 3 + 3] = v.copy()
      v = vv.copy()
    return inp, v, class_id


  def __len__(self):
    return self.nSamples

