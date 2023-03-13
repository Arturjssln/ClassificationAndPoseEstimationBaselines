import torch
import numpy as np
from utils.utils import AverageMeter, Flip
from utils.eval import AccViewCls
from utils.hmParser import parseHeatmap
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger
import torchmetrics

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  preds = []
  classification_acc = []
  Loss, Acc = AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('{}'.format(opt.expID), max=nIters)
  
  loss_func = torch.nn.CrossEntropyLoss()
  acc_func = torchmetrics.Accuracy().to(opt.GPU)
  save_classification_dict = {}
  save_classification_dict['pred'] = []
  save_classification_dict['true'] = []

  for i, (input, view, class_id) in enumerate(dataLoader):
    input_var = torch.autograd.Variable(input.cuda(opt.GPU)).float().cuda(opt.GPU)
    # import matplotlib.pyplot as plt 
    # print(input.shape, input.max(), input.min())
    target_var = torch.autograd.Variable(view.view(-1)).long().cuda(opt.GPU)
    class_id = torch.autograd.Variable(class_id.cuda(opt.GPU)).cuda(opt.GPU)
    output1, output2 = model(input_var)
    numBins = opt.numBins
    # TODO: why is there ignore_index = numBins ????
    loss1 = torch.nn.CrossEntropyLoss(ignore_index = numBins).cuda(opt.GPU)(output1.view(-1, numBins), target_var)
    loss2 = loss_func(output2, class_id).cuda(opt.GPU)
    loss = 0.5 * loss1 + 0.5 * loss2
    if split != 'train':
      pred_class = torch.max(output2, dim=1)[-1].item()
      true_class = class_id.item()

      save_classification_dict['pred'].append(pred_class)
      save_classification_dict['true'].append(true_class)

    pred_class = torch.max(output2, dim=1)[-1].cuda(opt.GPU)
    acc_current = acc_func(pred_class, class_id).item()
    # print('GT: ')
    # print(class_id)
    classification_acc.append(acc_current)

    Acc.update(AccViewCls(output1.data, view, numBins, opt.specificView))
    Loss.update(loss.item(), input.size(0))

    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      if opt.test:
        out = {}
        input_ = input.cpu().numpy()
        input_[0] = Flip(input_[0]).copy()
        inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], opt.inputRes, opt.inputRes)).float().cuda(opt.GPU)
        outputFlip = model(inputFlip_var)[0]
        pred = outputFlip.data.cpu().numpy()
        numBins = opt.numBins
        
        # if opt.specificView:
        #   nCat = len(ref.pascalClassId)
        #   pred = pred.reshape(1, nCat, 3 * numBins)
        #   azimuth = pred[0, :, :numBins]
        #   elevation = pred[0, :, numBins: numBins * 2]
        #   rotate = pred[0, :, numBins * 2: numBins * 3]
        #   azimuth = azimuth[:, ::-1]
        #   rotate = rotate[:, ::-1]
        #   output_flip = []
        #   for c in range(nCat):
        #     output_flip.append(np.array([azimuth[c], elevation[c], rotate[c]]).reshape(1, numBins * 3))
        #   output_flip = np.array(output_flip).reshape(1, nCat * 3 * numBins)
        # else:
        #   azimuth = pred[0][:numBins]
        #   elevation = pred[0][numBins: numBins * 2]
        #   rotate = pred[0][numBins * 2: numBins * 3]
        #   azimuth = azimuth[::-1]
        #   rotate = rotate[::-1]
        #   output_flip = np.array([azimuth, elevation, rotate]).reshape(1, numBins * 3)

        out['reg'] = output1.data.cpu().numpy()
        preds.append(out)
      
    final_classification_acc = sum(classification_acc) / len(classification_acc)
    Bar.suffix = '{split:5} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Pose Acc {Acc.avg:.6f} | Classification Acc {cls_acc:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, cls_acc = final_classification_acc)
    bar.next()
  bar.finish()
  return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds, save_classification_dict

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)

