import ref
import torch
import torch.nn as nn
import os
import torchvision.models as models
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
from models.hg import HourglassNet
from models.convnext import convnext
from models.resnetext import ResNetExt

# P3D+
NB_CLASSES = 12
# OOD 
# NB_CLASSES = 10

class Two_head_model(torch.nn.Module):
  def __init__(self, pretrained_model, final_layer_input_dim, final_layer_output_dim, dropout_p=None):
    super(Two_head_model, self).__init__()
    self.fc1 = nn.Linear(final_layer_input_dim, final_layer_output_dim)
    self.fc2 = nn.Linear(final_layer_input_dim, NB_CLASSES)
    self.model = pretrained_model
    self.with_dropout = dropout_p is not None
    if self.with_dropout:
      print(f"Using dropout, p = {dropout_p}")
      self.dropout = nn.Dropout(p=dropout_p)

  def forward(self, x):
    tmp = self.model(x)
    if self.with_dropout:
      tmp = self.dropout(tmp)
    out1 = self.fc1(tmp)
    out2 = self.fc2(tmp)
    return out1, out2


#Re-init optimizer
def getModel(opt): 
  if 'hg' in opt.arch:
    model = HourglassNet(opt.nStack, opt.nModules, opt.nFeats, opt.numOutput)
    optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                    alpha = ref.alpha, 
                                    eps = ref.epsilon, 
                                    weight_decay = ref.weightDecay, 
                                    momentum = ref.momentum)
  elif 'convnext' in opt.arch:
    model = convnext(pretrained=True, final_layer_input_dim=2048)
    model = Two_head_model(model, 2048, opt.numOutput, dropout_p=opt.dropout)
    optimizer = torch.optim.SGD(model.parameters(), opt.LR, momentum=0.9, weight_decay=1e-4)
  elif 'resnetext' in opt.arch:
    model = ResNetExt(pretrained=True, nb_classes=NB_CLASSES, nb_pose_classes=opt.numOutput)
    # model.freeze_extractor()
    optimizer = torch.optim.SGD(model.parameters(), opt.LR, momentum=0.9, weight_decay=1e-4)
  
  else:
    print("=> using pre-trained model '{}'".format(opt.arch))
    model = models.__dict__[opt.arch](weights="DEFAULT")
    if opt.arch.startswith('resnet'):
      model.avgpool = nn.AvgPool2d(8, stride=1)
      if '18' in opt.arch:
        model.fc = nn.Identity()
        model = Two_head_model(model, 512, opt.numOutput, dropout_p=opt.dropout)
      else :
        model.fc = nn.Identity()
        model = Two_head_model(model, 512 * 4, opt.numOutput, dropout_p=opt.dropout)
    if opt.arch.startswith('vit'):
      assert opt.arch in ['vit_b_16']
      model.heads = nn.Identity()
      model = Two_head_model(model, 768, opt.numOutput, dropout_p=opt.dropout)
    if opt.arch.startswith('swin_t'):
      model.head = nn.Identity()
      model = Two_head_model(model, 768, opt.numOutput, dropout_p=opt.dropout)

    if opt.arch.startswith('densenet'):
      if '161' in opt.arch:
        model.classifier = nn.Linear(2208, opt.numOutput)
      elif '201' in opt.arch:
        model.classifier = nn.Linear(1920, opt.numOutput)
      else:
        model.classifier = nn.Linear(1024, opt.numOutput)
    if opt.arch.startswith('vgg'):
      feature_model = list(model.classifier.children())
      feature_model.pop()
      feature_model.append(nn.Linear(4096, opt.numOutput))
      model.classifier = nn.Sequential(*feature_model)
    optimizer = torch.optim.SGD(model.parameters(), opt.LR,
                            momentum=0.9,
                            weight_decay=1e-4)
  
  if opt.loadModel != '':
    print("=> loading model '{}'".format(opt.loadModel))
    state_dict = torch.load(opt.loadModel, map_location="cuda:0")
    # if type(checkpoint) == type({}):
    #   state_dict = checkpoint['state_dict']
    # else:
    #   state_dict = checkpoint.state_dict()
    
    if 'hg' in opt.arch:
      for i in range(opt.nStack):
        if state_dict['tmpOut.{}.weight'.format(i)].size(0) != model.state_dict()['tmpOut.{}.weight'.format(i)].size(0):
          tmpOut = state_dict['tmpOut.{}.weight'.format(i)].clone()
          weightDim = tmpOut.size(0)
          state_dict['tmpOut.{}.weight'.format(i)] = torch.zeros(model.state_dict()['tmpOut.{}.weight'.format(i)].size())
          state_dict['tmpOut.{}.weight'.format(i)][:weightDim, :, :, :] = tmpOut.clone()[:weightDim, :, :, :]
          tmpOut_bias = state_dict['tmpOut.{}.bias'.format(i)].clone()
          state_dict['tmpOut.{}.bias'.format(i)] = torch.zeros(model.state_dict()['tmpOut.{}.bias'.format(i)].size())
          state_dict['tmpOut.{}.bias'.format(i)][:weightDim] = tmpOut_bias.clone()[:weightDim]

      for i in range(opt.nStack - 1):
        if state_dict['tmpOut_.{}.weight'.format(i)].size(1) != model.state_dict()['tmpOut_.{}.weight'.format(i)].size(1):
          tmpOut_ = state_dict['tmpOut_.{}.weight'.format(i)].clone()
          weightDim = tmpOut_.size(1)
          state_dict['tmpOut_.{}.weight'.format(i)] = torch.zeros(model.state_dict()['tmpOut_.{}.weight'.format(i)].size())
          state_dict['tmpOut_.{}.weight'.format(i)][:, :weightDim, :, :] = tmpOut_.clone()[:, :weightDim, :, :]

    model.load_state_dict(state_dict, strict=False)
  return model, optimizer
  
def saveModel(path, model, optimizer = None):
  if optimizer is None:
    torch.save({'state_dict': model.state_dict()}, path)
  else:
    torch.save({'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict()}, path)