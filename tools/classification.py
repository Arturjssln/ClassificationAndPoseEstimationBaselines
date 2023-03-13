from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import torch
import numpy as np
from datasets.Pascal3D import Pascal3D


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T


# def cls_func(score):
#     return np.argmax(np.sum(score, axis=1))


def cls_func(score):
    # [12, 3] -> [3]
    t = np.argmax(score, 0)

    # [3] -> [3, 12]
    s = convert_to_one_hot(t, 12)
    return np.argmax(np.sum(s, axis=1))


path = '../exp2/clsSpecTEST/preds.pth'
preds = torch.load(path)

opt = preds['opt']
preds = preds['preds']
n = len(preds)
dataset = Pascal3D(opt, 'val')

correct = 0
for idx in range(n):
    index = idx
    class_id = dataset.annot['class_id'][index]

    data = preds[index]['reg']
    data = data.reshape((12, 3, -1))

    # [12, 3]
    vote = np.max(data, axis=2)

    class_pred = cls_func(vote)

    if class_pred == class_id:
        correct += 1

print('Correct: %d/%d' % (correct, n))
print('Accuracy: %02f' % (correct / n))
