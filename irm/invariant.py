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

def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


def irm_train(args, model, device, train_loaders, optimizer, epoch):
  model.train()

  train_loaders = [iter(x) for x in train_loaders]

  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

  batch_idx = 0
  penalty_multiplier = epoch ** 1.6
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    for loader in train_loaders:
      data, target = next(loader, (None, None))
      if data is None:
        return loss_erm.mean()
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    (error + penalty_multiplier * penalty).backward()
    optimizer.step()

    if args.cges:
      if type(model) == nn.DataParallel:
        apply_cges(args, model.module, optimizer)
      else:
        apply_cges(args, model, optimizer)

    batch_idx += 1

def train_and_test_irm(args):

  import wandb

  wandb.init(entity="arjunashok", project="irm-notebook", config=vars(args))
  if not args.run_name:
    if args.cges:
        wandb.run.name = "irm-cges-"+str(args.lamb)
    else:
      wandb.run.name = "irm-plain"
  else:
    wandb.run.name = args.run_name

  set_seed(0)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
  train1_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

  train2_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train2',
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

  print(args)

  best_test_acc = 0

  for epoch in range(1, args.epochs):
    loss_erm = irm_train(args, model, device, [train1_loader, train2_loader], optimizer, epoch)
    train1_acc = test_model(model, device, train1_loader, set_name='train1_set', epoch=epoch)
    train2_acc = test_model(model, device, train2_loader, set_name='train2_set', epoch=epoch)
    test_acc = test_model(model, device, test_loader, epoch=epoch)

    if test_acc > best_test_acc:
      best_test_acc = test_acc

    lr_scheduler.step(epoch)

    wandb.log({"train/loss": loss_erm.item()}, step=epoch)
    wandb.log({"test_set/best_acc": best_test_acc}, step=epoch)

    # if train1_acc > 70 and train2_acc > 70 and test_acc > 60:
    #   print('found acceptable values. stopping training.')
    #   return