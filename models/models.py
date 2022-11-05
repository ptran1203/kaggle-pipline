import torch
import torch.nn as nn
import timm


def do_freeze_bn(model): 
    c = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            c += 1
            module.eval()

    print(f"Freeze {c} batchnorm layers")


class Model(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b3_ns', pretrained=False, freeze_bn=False, num_classes=1):
        super(Model, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)

        if freeze_bn:
            do_freeze_bn(self.backbone)
        if 'efficientnet' in backbone:
            self.in_features = self.backbone.classifier.in_features
        elif 'resne' in backbone: # Resnet family
            self.in_features = self.backbone.fc.in_features
        elif 'senet' in backbone:
            self.in_features = self.last_linear.in_features
        else:
            raise ValueError(backbone)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.backbone.forward_features(x)
        features = self.pooling(features)
        features = features.view(batch_size, -1)
        out = self.head(features)
        return out
