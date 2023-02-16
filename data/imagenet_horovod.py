import os
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.data_utils import SubsetDistributedSampler
from data.data_utils import ImageNetPolicy, Cutout

import horovod.torch as hvd

class Data:
    def __init__(self, args):
        pin_memory = False
        if args.gpus is not None:
            pin_memory = True

        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if args.autoaug==True:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]))

        if args.cutout==True:
            trainset.transforms.append(Cutout(args.cutout_length))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())

        self.testLoader = DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            sampler=self.test_sampler,
            num_workers=4,
            pin_memory=True)