import torch
import torch.nn as nn
import torchvision.models as models

from .layers.upsampling import DoubleConv
from .layers.upsampling import Up

class ResNetExt(nn.Module):
    def __init__(self, pretrained, nb_classes, nb_pose_classes, type="classifiers"):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4
        
        self.classifier = nn.Sequential()
        self.classifier.add_module("0", net.avgpool)
        self.classifier.add_module("1", nn.Flatten())
        self.classifier.add_module("2", nn.Linear(2048, 1024, bias=True))
        self.classifier.add_module("3", nn.ReLU(inplace=True))
        self.classifier.add_module("4", nn.Linear(1024, 512, bias=True))
        self.classifier.add_module("5", nn.ReLU(inplace=True))
        
        self.head1 = nn.Linear(512, nb_pose_classes, bias=True)
        self.head2 = nn.Linear(512, nb_classes, bias=True)

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = Up(1024, 512, 256)
        
        assert type in ["all", "classifiers", "nemo"], f"Unnkown type: {type}"
        self.type_ = type
            

    def freeze_extractor(self):
        for param in self.extractor.parameters():
            param.requires_grad = False
        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def _feature_extraction(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        return x1, x2, x3
        
    def _forward_classifiers(self, x):
        f2 = self.classifier(x)
        h1 = self.head1(f2)
        h2 = self.head2(f2)
        return h1, h2
        
    def _forward_nemo(self, x1, x2, x3):
        x4 = self.upsample0(x3)
        x5 = self.upsample1(x4, x2)
        x6 = self.upsample2(x5, x1)
        return x6
    
    def forward(self, x):
        if self.type_ == "all":
            return self.forward_all(x)
        if self.type_ == "classifiers":
            return self.forward_classifiers(x)
        if self.type_ == "nemo":
            return self.forward_nemo(x)
        
    
    def forward_all(self, x):
        x1, x2, x3 = self._feature_extraction(x)
        h1, h2 = self._forward_classifiers(x3)
        x6 = self._forward_nemo(x1, x2, x3)
        return x6, h1, h2
    
    def forward_classifiers(self, x):
        *_, x3 = self._feature_extraction(x)
        h1, h2 = self._forward_classifiers(x3)
        return h1, h2
    
    def forward_nemo(self, x):
        x1, x2, x3 = self._feature_extraction(x)
        x6 = self._forward_nemo(x1, x2, x3)
        return x6