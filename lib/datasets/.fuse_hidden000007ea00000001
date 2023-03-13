import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip
from utils.img import Crop, DrawGaussian, Transform, Transform3D
import os

# occ_level = 'FGL3_BGL3'
# occ_level = ''

using_occ_images_during_training = False


class Pascal3D(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing pascal3d {} data.'.format(split))
    annot = {}
    tags = ['bbox', 'anchors', 'vis', 'dataset', 'class_id', 'imgname', 
            'viewpoint_azimuth', 'viewpoint_elevation', 'viewpoint_theta', 'anchors_3d', 
            'space_embedding', 'truncated', 'occluded', 'difficult', 'valid', 'cad_index']
    if len(opt.occ) == 0:
      self.occ_level = ''
    else:
      self.occ_level = 'FG%s_BG%s' % (opt.occ, opt.occ)

    if len(self.occ_level) > 0:
        f = File('{}/Pascal3D/Pascal3D_{}-{}.h5'.format(ref.dataDir, self.occ_level, split), 'r')
    else:
        f = File('{}/Pascal3D/Pascal3D-{}.h5'.format(ref.dataDir, split), 'r')

    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    annot['index'] = np.arange(len(annot['class_id']))
    tags = tags + ['index']
    
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
    self.nSamples = len(annot['vis'])

    self.predicted_2d = np.load(opt.load_anno_path_kp)
    self.predicted_R = np.load(opt.load_anno_path_R)

    print 'Loaded Pascal3D {} {} samples'.format(split, self.nSamples)

  def LoadImage(self, index):
    img_name = ''
    for v in range(len(self.annot['imgname'][index])):
      c = self.annot['imgname'][index][v]
      if c != 0:
        img_name += chr(c)

    occ_level = self.occ_level
    if len(occ_level) == 0 or self.split == 'train':
      flag = True
      if self.split == 'train' and using_occ_images_during_training and np.random.rand() < 0.5:
        path = '/home/angtian/angtian/workspace/PASCAL3D+/PASCAL_NEWs_TRAIN/images/{}_occluded/{}'.format(ref.pascalClassName[self.annot['class_id'][index]], img_name)
        flag = False
        if not os.path.exists(path):
          flag = True
      if flag:
        path = '{}/Images/{}_{}/{}'.format(ref.pascal3dDir, ref.pascalClassName[self.annot['class_id'][index]], 
                                    ref.pascalDatasetName[self.annot['dataset'][index]], img_name)
    else:
        path = '/home/angtian/angtian/workspace/PASCAL3D+/PASCAL_NEWs/images/{}{}/{}'.format(ref.pascalClassName[self.annot['class_id'][index]], occ_level, img_name)
    # print(path)
    img = cv2.imread(path)
    return img, img_name


  def GetPartInfo(self, index, img_name):
    # pts2d = self.annot['anchors'][index].copy()
    # pts3d = self.annot['anchors_3d'][index].copy()
    # emb = self.annot['space_embedding'][index].copy()

    emb = model_pts[self.annot['cad_index'][index] - 1].copy()
    pts2d = self.predicted_2d[img_name.split('.')[0]].copy()

    # Put theta at R[3, 3]
    pts3d = get_p3d(emb, self.predicted_R[img_name.split('.')[0]], self.predicted_R[img_name.split('.')[0]][3, 3], self.annot['principal'][index])

    box = self.annot['bbox'][index].copy()
    c = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    s = max((box[2] - box[0]), (box[3] - box[1])) * ref.padScale
    return pts2d, pts3d, emb, c, s

  def __getitem__(self, index):
    img, img_name = self.LoadImage(index)
    pts2d, pts3d, emb, c, s = self.GetPartInfo(index, img_name)
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    pts3d[:, 2] += s / 2

    r = 0
    if self.split == 'train':
      s = s * (2 ** Rnd(ref.scale))
      c[1] = c[1] + Rnd(ref.shiftY)
      r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
    inp = Crop(img, c, s, r, ref.inputRes)
    inp = inp.transpose(2, 0, 1).astype(np.float32) / 256.
    
    starMap = np.zeros((1, ref.outputRes, ref.outputRes))
    embMap = np.zeros((3, ref.outputRes, ref.outputRes))
    depMap = np.zeros((1, ref.outputRes, ref.outputRes))
    mask = np.concatenate([np.ones((1, ref.outputRes, ref.outputRes)), np.zeros((4, ref.outputRes, ref.outputRes))]);

    for i in range(pts3d.shape[0]):
      if self.annot['valid'][index][i] > ref.eps:
        if (self.annot['vis'][index][i] > ref.eps):
          pt3d = Transform3D(pts3d[i], c, s, r, ref.outputRes).astype(np.int32)
          pt2d = Transform(pts2d[i], c, s, r, ref.outputRes).astype(np.int32)
          if pt2d[0] >= 0 and pt2d[0] < ref.outputRes and pt2d[1] >=0 and pt2d[1] < ref.outputRes:
            embMap[:, pt2d[1], pt2d[0]] = emb[i]
            depMap[0, pt2d[1], pt2d[0]] = 1.0 * pt3d[2] / ref.outputRes - 0.5
            mask[1:, pt2d[1], pt2d[0]] = 1
          starMap[0] = np.maximum(starMap[0], DrawGaussian(np.zeros((ref.outputRes, ref.outputRes)), pt2d, ref.hmGauss).copy())

    out = starMap
    if 'emb' in self.opt.task:
      out = np.concatenate([out, embMap])
    if 'dep' in self.opt.task:
      out = np.concatenate([out, depMap])
    mask = mask[:out.shape[0]].copy()

    if self.split == 'train':
      if np.random.random() < 0.5:
        inp = Flip(inp)
        out = Flip(out)
        mask = Flip(mask)
        if 'emb' in self.opt.task:
          out[1] = - out[1]
    return inp, out, mask

  def __len__(self):
    return self.nSamples


def get_p3d(p3d, R, theta, principal, viewport=3000, f=1):
    M = viewport
    # P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];
    P_ = np.array([[M * f, 0, 0], [0, M * f, 0], [0, 0, -1]])

    # [3, 3] @ [3, 3 + 1] -> [3, 4]
    P = np.dot(P_, R[0:3, 0:4])

    x3d = p3d
    # [3, 4] @ [n, 3 + 1].T -> [3, n]
    x3d = np.dot(P, np.concatenate([x3d, np.ones((x3d.shape[0], 1))], axis = 1).transpose())

    x = x3d.copy()
    x[2, x[2, :] == 0] = 1
    x[0] = x[0] / x[2]
    x[1] = x[1] / x[2]
    x = x[:2]

    R2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.dot(R2d, x).transpose()
    x[:, 1] = - x[:, 1]
    x[:, 0] += principal[0]
    x[:, 1] += principal[1]

    p3d = np.dot(R[0:3, 0:3], p3d.transpose()).transpose()
    p3d[:, :2] = np.dot(R2d, p3d[:, :2].transpose()).transpose()
    p3d[:, 1] = - p3d[:, 1]
    mean_p = p3d.mean(axis=0)
    std_p = max(p3d[:, 0].max() - p3d[:, 0].min(), p3d[:, 1].max() - p3d[:, 1].min())
    mean_x = x.mean(axis=0)
    std_x = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() - x[:, 1].min())
    for j in range(p3d.shape[0]):
        p3d[j, 0] = (p3d[j, 0] - mean_p[0]) / std_p * std_x + mean_x[0]
        p3d[j, 1] = (p3d[j, 1] - mean_p[1]) / std_p * std_x + mean_x[1]
        p3d[j, 2] = (p3d[j, 2] - mean_p[2]) / std_p * std_x
    return p3d
