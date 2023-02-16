import os
import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(2019)
import random

random.seed(2019)
from torch.utils.data.sampler import SubsetRandomSampler
import pickle

class Data:
    def __init__(self, args):
        pin_memory = True
        valid_size = 5000

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

        trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)

        num_train = len(trainset)
        indices = list(range(num_train))
        # np.random.shuffle(indices)
        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory,sampler=train_sampler
        )

        self.vaildLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory,sampler=valid_sampler
        )

        testset = CIFAR10(root=args.data_path, train=False, download=False, transform=transform_test)

        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)