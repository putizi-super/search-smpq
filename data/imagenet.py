import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data.data_utils import SubsetDistributedSampler
from data.data_utils import ImageNetPolicy, Cutout
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
            train_compose = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            train_compose = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])

        if args.cutout==True:
            train_compose.transforms.append(Cutout(args.cutout_length))

        trainset = datasets.ImageFolder(
            traindir,
            train_compose
        )

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            # num_workers=args.workers,
            num_workers=0,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testLoader = DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            # num_workers=args.workers,
            pin_memory=True)