import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes: int):
        super(CNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

