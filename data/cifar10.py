from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.data_utils import SubsetDistributedSampler
from data.data_utils import CIFAR10Policy, Cutout

class Data:
    def __init__(self, args):
        pin_memory = True
        CIFAR_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
        CIFAR_STD = [x / 255 for x in [63.0, 62.1, 66.7]]
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
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)

        self.trainLoader = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory
        )

        testset = CIFAR10(root=args.data_path, train=False, download=False, transform=transform_test)
        self.testLoader = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=2, pin_memory=pin_memory)