"""Reconstruct the head defined in both existing ResNet training notebooks."""
from torch import nn
from torchvision.models import resnet18


def build_classifier(num_classes):
    model = resnet18(weights=None)  # No downloads or pretrained-weight replacement.
    model.fc = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model
