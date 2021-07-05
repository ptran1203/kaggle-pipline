import timm
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', n_classes=2):
        super().__init__()
        self.name = model_name
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output
