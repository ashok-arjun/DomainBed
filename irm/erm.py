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
from argparse import Namespace
import wandb

from utils import set_seed
from colored_mnist import ColoredMNIST
from models import Net, ConvNet
from train_test import test_model
from cges_utils import apply_cges

def erm_train(args, model, device, train_loader, optimizer, epoch):
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device).float()
    optimizer.zero_grad()
    output = model(data)
    loss = F.binary_cross_entropy_with_logits(output, target)
    loss.backward()
    optimizer.step()

    if args.cges:
        if type(model) == nn.DataParallel:
          apply_cges(args, model.module, optimizer)
        else:
          apply_cges(args, model, optimizer)

  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
  
  wandb.log({"train/loss": loss.item()}, step=epoch)


def train_and_test_erm(args):
  wandb.init(entity="arjunashok", project="irm-notebook", config=vars(args))
  if not args.run_name:
    if args.cges:
        wandb.run.name = "erm-cges-"+str(args.lamb)
    else:
      wandb.run.name = "erm-plain"
  else:
    wandb.run.name = args.run_name

  set_seed(0)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
  all_train_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True, **kwargs)

  model = ConvNet()
  
  if torch.cuda.device_count() > 1:
    print("Data Parallel!")
    model = nn.DataParallel(model)

  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

  lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.05)

  best_test_acc = 0

  for epoch in range(1, args.epochs):
    erm_train(args, model, device, all_train_loader, optimizer, epoch)
    test_model(model, device, all_train_loader, set_name='train_set', epoch=epoch)
    test_acc = test_model(model, device, test_loader, epoch=epoch)

    if test_acc > best_test_acc:
      best_test_acc = test_acc

    lr_scheduler.step(epoch=epoch)

    wandb.log({"test_set/best_acc": best_test_acc}, step=epoch)