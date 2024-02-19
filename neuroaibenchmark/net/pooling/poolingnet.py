"""
The CORnet models provided contain a mix of convolutions, recurrent computations, and pooling layers. If we are to design PyTorch models that use only pooling for classification and match the variations in the CORnet models, we need to interpret the CORnet architecture to a purely pooling-based design. This means we will be removing the convolutional components and focusing solely on pooling operations to downsample the image while preserving the spatial hierarchy of the original models.

Given the constraint of using only pooling layers for classification, we might lose the ability to capture the hierarchical feature representations that convolutions provide. Nevertheless, for the sake of the exercise, we can create variations of pooling models. Here's how we could translate the four CORnet variations into pooling-only models:

1. Pooling Model Matching CORblock_Z:
    This model will use max pooling to reduce the spatial dimensions, followed by an adaptive average pooling to prepare for the classifier.

2. Pooling Model Matching CORblock_R:
    Since CORblock_R has a recurrent structure, we could interpret this as repeatedly applying the same pooling operation, possibly with a skip connection that adds the input of the block to its output at each time step.

3. Pooling Model Matching CORblock_RT:
    Similar to CORblock_R, but since RT might imply real-time or recurrent with a twist, we could use different pooling strategies at each time step.

4. Pooling Model Matching CORblock_S:
    This block uses a scale factor to imply a bottleneck architecture. We could simulate this by varying the pooling window and stride to aggressively reduce dimensions and then use adaptive pooling to match the output size.
"""

import torch
import torch.nn as nn
import math
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PoolingNet_Z(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(3 * 3 * 64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PoolingNet_R(nn.Module):
    def __init__(self, num_classes=1000, times=5):
        super().__init__()
        self.times = times
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(56 * 56 * 64, num_classes)

    def forward(self, x):
        for _ in range(self.times):
            x = self.pooling(x) + x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PoolingNet_RT(nn.Module):
    def __init__(self, num_classes=1000, times=5):
        super().__init__()
        self.times = times
        self.pooling = nn.ModuleList([nn.MaxPool2d(kernel_size=3, stride=2, padding=1) for _ in range(times)])
        self.classifier = nn.Linear(56 * 56 * 64, num_classes)

    def forward(self, x):
        for pool in self.pooling:
            x = pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PoolingNet_S(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        scale = 4
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64 * scale * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
