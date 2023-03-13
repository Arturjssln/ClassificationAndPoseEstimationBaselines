from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
import datetime
from utils.logger import Logger
import torch
import torch.utils.data
from opts import opts
import ref
from model import getModel, saveModel
opt = opts().parse()
import json
from datasets.Pascal3DCls import Pascal3DOOD as Dataset
from trainCls import train, val


def main():
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))
  model, optimizer = getModel(opt)

  criterion = torch.nn.MSELoss()
  
  if opt.GPU > -1:
    print('Using GPU', opt.GPU)
    model = model.cuda(opt.GPU)
    criterion = criterion.cuda(opt.GPU)
  
  

  if opt.test:
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val', nuisance=opt.nuisance), 
      batch_size = 1, 
      shuffle = True if opt.DEBUG > 1 else False,
      num_workers = 1
    )
    _, preds, save_classification_dict = val(0, opt, val_loader, model, criterion)
    if opt.nuisance is not None:
      torch.save({'opt': opt, 'preds': preds}, os.path.join(opt.saveDir, 'pose_preds_%s.pth' % opt.nuisance))
      f = open(os.path.join(opt.saveDir, 'classification_preds_%s.json' % opt.nuisance), 'w')
      json.dump(save_classification_dict, f)
    else:
      raise ValueError('Nuisance is None')

    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train', nuisance=opt.nuisance), 
      batch_size = opt.trainBatch, 
      shuffle = True,
      num_workers = 1
      # num_workers = int(opt.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    mark = epoch if opt.saveAllModels else 'last'
    log_dict_train, _ , _= train(epoch, opt, train_loader, model, criterion, optimizer)
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    '''
    if epoch % opt.valIntervals == 0:
      log_dict_val, preds, save_classification_dict = val(epoch, opt, val_loader, model, criterion)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      saveModel(os.path.join(opt.saveDir, 'model_{}.checkpoint'.format(mark)), model) # optimizer
    logger.write('\n')
    '''
    if epoch % opt.dropLR == 0:
      lr = opt.LR * (0.1 ** (epoch // opt.dropLR))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()
  torch.save(model.cpu().state_dict(), os.path.join(opt.saveDir, 'model_cpu.pth'))

if __name__ == '__main__':
  main()
