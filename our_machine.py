import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights


class OurMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        number_input = self.resnet.fc.out_features
        self.fc = nn.Sequential(
            nn.Linear(number_input, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2)
        )

        for param in self.resnet.layer1.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
