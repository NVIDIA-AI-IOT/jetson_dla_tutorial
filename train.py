# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models import MODELS


parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='The name of the model.  See the MODELS dictionary in models.py for options.')
parser.add_argument('--batch_size', type=int, default=64, help='The data loader batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='The optimizer learning rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer type.  Must be one of the keys in the OPTIMIZERS variable in train.py.')
parser.add_argument('--momentum', type=float, default=0.9, help='The optimizier momentum.  Only applies when optimizer=sgd.')
parser.add_argument('--epochs', type=int, default=50, help='The number of training epochs.')
parser.add_argument('--dataset_path', type=str, default='data/cifar10', help='The directory to store generated models and logs.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to store the model weights.')
args = parser.parse_args()

OPTIMIZERS = {
    'sgd': lambda params, args: torch.optim.SGD(params, lr=args.lr, momentum=args.momentum),
    'adam': lambda params, args: torch.optim.Adam(params, lr=args.lr)
}

model = MODELS[args.model_name]().cuda()
optimizer = OPTIMIZERS[args.optimizer](model.parameters(), args)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, 
    train=True,
    download=True, 
    transform=transform_train
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=args.batch_size,
    shuffle=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, 
    train=False,
    download=True, 
    transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=args.batch_size,
    shuffle=False
)

best_accuracy = 0.0

for epoch in range(args.epochs):

    train_loss = 0.0
    test_loss = 0.0
    train_accuracy = 0
    test_accuracy = 0

    # train loop

    model = model.train()

    for image, label in iter(train_loader):
        
        optimizer.zero_grad()

        image = image.cuda()
        label = label.cuda()

        output = model(image)

        loss = F.cross_entropy(output, label)

        loss.backward()
        optimizer.step()

        train_loss += float(loss)
        train_accuracy += int(torch.sum(output.argmax(dim=-1) == label))

    train_accuracy /= len(train_dataset)
    train_loss /= len(train_loader)

    model = model.eval()

    for image, label in iter(test_loader):

        image = image.cuda()
        label = label.cuda()

        output = model(image)

        loss = F.cross_entropy(output, label)

        test_loss += float(loss)
        test_accuracy += int(torch.sum(output.argmax(dim=-1) == label))

    test_accuracy /= len(test_dataset)
    test_loss /= len(test_loader)

    print(f'{epoch}, {train_loss}, {test_loss}, {train_accuracy}, {test_accuracy}')

    if test_accuracy > best_accuracy and args.checkpoint_path is not None:
        print(f'Saving checkpoint to {args.checkpoint_path} for model with test accuracy {test_accuracy}.')
        torch.save(model.state_dict(), args.checkpoint_path)