import torch
import torch.nn as nn
import torchvision.models as models

class DinoViT(nn.Module):
    def __init__(self, pretrained, patch_size=16, size="small"):
        super().__init__()
        assert patch_size in [8, 16], "Patch size must be 8 or 16"
        assert size in ["small", "base"], "Size must be small or base"
        self.patch_size = patch_size
        self.size = "s" if size == "small" else "b"
        self.net = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vit" + self.size + str(self.patch_size),
            pretrained=pretrained,
        )

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.net.prepare_tokens(x)
        for _, blk in enumerate(self.net.blocks):
            x = blk(x)
        x = self.net.norm(x)
        x = x.permute(0, 2, 1)[..., 1:].reshape(b, -1)
        
        return x
