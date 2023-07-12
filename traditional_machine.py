import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights


class TraditionalMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(number_input, 2)

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)
