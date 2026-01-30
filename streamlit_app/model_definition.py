import torch.nn as nn
from torchvision import models


def build_model(num_classes=2):
    """
    Builds the exact ResNet50 architecture used during training.
    Must match the notebook definition 1:1.
    """
    model = models.resnet50(weights=None)

    # Freeze backbone (optional for inference, but matches training intent)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
