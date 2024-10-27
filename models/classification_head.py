import torch.nn as nn

from .resnet import resnet50

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class Classification(nn.Module):
    def __init__(self, name, num_classes=1):
        super(Classification, self).__init__()
        self.fc = nn.Linear( CHANNELS[name], num_classes )
        # self.model = resnet50(pretrained=True)

    def forward(self, x, return_feature=False):
        return self.fc(x)
