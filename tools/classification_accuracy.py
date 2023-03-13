import json
import os
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Classification accuaracy calculation')
parser.add_argument('--load_dir', default='', type=str)
args = parser.parse_args()

CLASS_NAMES = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]

with open(args.load_dir, 'r') as f:
    tmp = json.load(f)
y_pred = tmp['pred']
y_true = tmp['true']
conf_matrix = confusion_matrix(y_pred, y_true)
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
    acc = pred_count[i] / ground_truth_count[i]
    print('The accuaracy of class ' + CLASS_NAMES[i] + ' is ' + str(acc))

acc = sum(pred_count) / len(y_true)
print('The aggregated accuaracy is ' + str(acc))



