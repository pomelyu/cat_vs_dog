import torch
from torchvision.models import resnet34

from .basic_module import BasicModule

class ResNet34(BasicModule):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model_name = "resnet34"
        self.model = resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 1, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        x = x.view(-1)
        return x
