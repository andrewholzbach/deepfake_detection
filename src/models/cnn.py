import torch.nn as nn
from torchvision import models

# Load a pre-trained ResNet model
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained ResNet, remove the last layer
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the classification layer
        # Add a fully connected layer to adjust feature vector size
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x  # This is the feature vector
