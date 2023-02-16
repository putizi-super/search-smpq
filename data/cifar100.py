from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.data_utils import SubsetDistributedSampler
from data.data_utils import CIFAR10Policy, Cutout

class Data:
    def __init__(self, args):
        pin_memory = True
        CIFAR_MEAN = [x / 255 for x in [129.3, 124.1, 112.4]]
        CIFAR_STD = [x / 255 for x in [68.2, 65.4, 70.4]]
        if args.autoaug==True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=128),
                transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])    
        else:        
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

        if args.cutout==True:
            transform_train.transforms.append(Cutout(args.cutout_length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)

        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=2, pin_memory=pin_memory
        )

        testset = CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)

# Data for Search (trainLoader shuffle=False)
class Data_Search:
    def __init__(self, args):
        pin_memory = True

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

        trainset = CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)

        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory
        )

        testset = CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)