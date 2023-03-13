from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import sys
import numpy as np
import cv2
import ref
import torch
from datasets.Pascal3DCls import Pascal3DOOD as Pascal3D
from utils.debugger import Debugger
from utils.hmParser import parseHeatmap
from utils.horn87 import RotMat, horn87
from scipy.linalg import logm
PI = np.arccos(-1)
DEBUG = False
import json
import os
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Classification accuaracy calculation')
parser.add_argument('--load_classification_dir', default='', type=str)
parser.add_argument('--load_pred_dir', default='', type=str)
parser.add_argument('--nuisance', default=None, type=str)
args = parser.parse_args()
# # P3D+
# CLASS_NAMES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
# OOD
CLASS_NAMES = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]


def angle2dcm(angle):
  azimuth = angle[0]
  elevation = angle[1]
  theta = angle[2]
  return np.dot(RotMat('Z', theta), np.dot(RotMat('X', - (PI / 2 - elevation)), RotMat('Z', - azimuth)))
  
def Rotate(points, angle):
  azimuth = angle[0]
  elevation = angle[1]
  theta = angle[2]
  azimuth = - azimuth
  elevation = - (PI / 2 - elevation)
  Ra = RotMat('Z', azimuth)
  Re = RotMat('X', elevation)
  Rt = RotMat('Z', theta)
  ret = np.dot(np.dot(Rt, np.dot(Re, Ra)), points.transpose()).transpose()
  ret[:, 1] *= -1
  return ret
  
with open(args.load_classification_dir, 'r') as f:
    tmp = json.load(f)
y_pred = tmp['pred']
y_true = tmp['true']
conf_matrix = confusion_matrix(y_pred, y_true)


path = args.load_pred_dir
preds = torch.load(path)
opt = preds['opt']
preds = preds['preds']
n = len(preds)

num = {}
acc = {}
acc10 = {}
err = {}
pi_over_6 = [0 for i in range(len(CLASS_NAMES))]
pi_over_18 = [0 for i in range(len(CLASS_NAMES))]
for v in CLASS_NAMES:
  acc[v], num[v] = 0, 0
  acc10[v] = 0
  err[v] = []
  
dataset = Pascal3D(opt, 'val', nuisance=args.nuisance)
print('Evaluating...')
for idx in range(n):
  index = idx if not DEBUG else np.random.randint(n)
  
  class_id = dataset.annot['class_id'][index]
  class_name = CLASS_NAMES[class_id]
  v = np.array([dataset.annot['viewpoint_azimuth'][index], dataset.annot['viewpoint_elevation'][index], dataset.annot['viewpoint_theta'][index]]) / 180.
  gt_view = v * PI
  output = preds[index]['reg']
  numBins = opt.numBins
  binSize = 360. / opt.numBins
  current_label = y_true[index]
  current_pred = y_pred[index]
  
  try:
    _, pred = torch.from_numpy(output).view(3, numBins).topk(1, 1, True, True)
  except:
    _, pred = torch.from_numpy(output[0])[class_id * 3 * numBins:(class_id + 1) * 3 * numBins].view(3, numBins).topk(1, 1, True, True)
  #https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/10fe7c7a74b3369dce9a3a13b3a7f85af859435b/utils/poseHypotheses.m#L53
  pred = (pred.view(3).float()).numpy()
  pred[0] = (pred[0] + 0.5) * PI / (opt.numBins / 2.)
  pred[1] = (pred[1] - opt.numBins / 2) * PI / (opt.numBins / 2.)
  pred[2] = (pred[2] - opt.numBins / 2) * PI / (opt.numBins / 2.)

  bestR = angle2dcm(pred)
  
  R_gt = angle2dcm(gt_view)
  err_ = ((logm(np.dot(np.transpose(bestR), R_gt)) ** 2).sum()) ** 0.5 / (2.**0.5) * 180 / PI

  num[class_name] += 1
  acc[class_name] += 1 if err_ <= 30. else 0
  acc10[class_name] += 1 if err_ <= 10. else 0
  err[class_name].append(err_)

  pi_over_6[CLASS_NAMES.index(class_name)] += 1 if (err_ <= 30. and current_label == current_pred) else 0
  pi_over_18[CLASS_NAMES.index(class_name)] += 1 if (err_ <= 10. and current_label == current_pred) else 0

  if DEBUG:
    input, target, mask, view = dataset[index]
    debugger = Debugger()
    img = (input[:3].transpose(1, 2, 0)*256).astype(np.uint8).copy()
    debugger.addImg(img)
    debugger.showAllImg(pause = False)


accAll = 0.
acc10All = 0.
numAll = 0.
mid = {}
err_all = []
for v in CLASS_NAMES:
  accAll += acc[v]
  acc10All += acc10[v]
  numAll += num[v]
  if num[v] == 0:
    acc[v] = 0.0
    acc[v] = 0.0
    mid[v] = 0.0
    error_all = err_all + err[v]
    continue
  acc[v] = 1.0 * acc[v] / num[v]
  acc10[v] = 1.0 * acc10[v] / num[v]
  mid[v] = np.sort(np.array(err[v]))[len(err[v]) // 2] 
  err_all = err_all + err[v]

cates = CLASS_NAMES
print(cates)
print('num', [num[t] for t in cates])
print('Acc30', [acc[t] for t in cates])
print('Acc10', [acc10[t] for t in cates])
print('mid', [mid[t] for t in cates])
print('acc30All', accAll / numAll)
print('acc10All', acc10All / numAll)
print('midAll', np.sort(np.array(err_all))[len(err_all) // 2]) 

print(conf_matrix)

ground_truth_count = list(0 for i in range(len(CLASS_NAMES)))
pred_count = list(0 for i in range(len(CLASS_NAMES)))

for i in range(len(y_pred)):
    pred = y_pred[i]
    truth = y_true[i]
    ground_truth_count[truth] += 1
    if pred == truth:
        pred_count[truth] += 1

for i in range(len(CLASS_NAMES)):
    if ground_truth_count[i] == 0:
      continue
    acc = pred_count[i] / ground_truth_count[i]
    print('The accuaracy of class ' + CLASS_NAMES[i] + ' is ' + str(acc))

acc = sum(pred_count) / len(y_true)
print('The aggregated accuaracy is ' + str(acc))

error1 = (sum(pi_over_6)) / len(y_true)
print('Aggregated portion of images rightly classified with error under pi/6 is ' + str(error1))

error2 = (sum(pi_over_18)) / len(y_true)
print('Aggregated portion of images rightly classified with error under pi/18 is ' + str(error2))




