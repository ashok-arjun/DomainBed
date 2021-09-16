import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils


class Net(nn.Module):
  def __init__(self, **kwargs):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(3 * 28 * 28, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = x.view(-1, 3 * 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


class ConvNet(nn.Module):
  def __init__(self, **kwargs):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    logits = self.fc2(x).flatten()
    return logits

class SimpleModel(nn.Module):
    def __init__(self, num_channels=[64, 128, 256], batch_norm=True, num_classes=10, **kwargs):
        super().__init__()

        layers = []

        num_channels = [3] + num_channels

        for i in range(1, len(num_channels)):
            layers.extend(self._make_layer(num_channels[i-1], num_channels[i], batch_norm))

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(num_channels[-1] * 2 * 2, num_classes)


    def _make_layer(self, input_channels, output_channels, batch_norm=True):
        layer = []

        layer.append(nn.Conv2d(input_channels, output_channels, 3, 1, 1))
        if batch_norm:
            layer.append(nn.BatchNorm2d(output_channels))
        layer.append(nn.ReLU())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return layer

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x