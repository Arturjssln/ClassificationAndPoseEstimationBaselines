import torch.nn as nn
import torchvision.models as models

def convnext(pretrained, final_layer_input_dim):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier = model.classifier[:-1]  # remove final layer
    model.classifier.add_module("2", nn.Linear(768, final_layer_input_dim, bias=True))
    return model

