import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import scale
import random
import os
import numbers
from PIL import ImageFilter
import logging

SVHN_mean = tuple([x / 255 for x in [129.3, 124.1, 112.4]])
SVHN_std = tuple([x / 255 for x in [68.2, 65.4, 70.4]])
MNIST_mean = (0,)
MNIST_std = (1,)
CIFAR10_mean = (0.4914, 0.4822, 0.4465)
CIFAR10_std = (0.2023, 0.1994, 0.2010)


def get_train_loaders(args):
    assert args.dataset_size > 0 and args.dataset_size <= 1.
    assert args.valid_portion >= 0 and args.valid_portion < 1.
    train_data = None
    if args.dataset == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor(), \
                                transforms.Normalize(MNIST_mean, MNIST_std)])

        train_data = datasets.MNIST(root=args.data, train=True, 
                                    download=True, transform=train_transform)
    elif args.dataset == "cifar":
        train_transform =[]
        if not args.q: 
            train_transform.append(transforms.RandomCrop(32, padding=4))
            train_transform.append(transforms.RandomHorizontalFlip())

        train_transform.append(transforms.ToTensor())
        train_transform.append(transforms.Normalize(CIFAR10_mean, CIFAR10_std))
        
        if not args.q:
            train_transform.append(transforms.RandomErasing())

        train_transform = transforms.Compose(train_transform)

        train_data = datasets.CIFAR10(root=args.data, train=True, 
                                    download=True, transform=train_transform)
    elif args.dataset == "svhn":
        train_transform = []
        if not args.q:
            train_transform.append(transforms.RandomCrop(32, padding=4))
            train_transform.append(transforms.RandomHorizontalFlip())

        train_transform.append(transforms.ToTensor())
        train_transform.append(transforms.Normalize(
            SVHN_mean, SVHN_std))

        if not args.q:
            train_transform.append(transforms.RandomErasing())

        train_transform = transforms.Compose(train_transform)

        train_data = datasets.SVHN(root=args.data, split='train',
                                      download=True, transform=train_transform)
    else:
        raise NotImplementedError("Other datasets not implemented")
    return get_train_split_loaders(args.dataset_size, args.valid_portion, train_data, args.batch_size, args.data, 0)



def get_train_split_loaders(dataset_size, valid_portion, train_data, batch_size, path_to_save_data, num_workers=0):
    num_train = int(np.floor(len(train_data) * dataset_size))
    indices = list(range(len(train_data)))
    indices = random.sample(indices, num_train)
    valid_split = int(
        np.floor((valid_portion) * num_train))   # 40k
    valid_idx, train_idx = indices[:valid_split], indices[valid_split:]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=num_workers)
    
    valid_loader = None
    if valid_portion>0.0:
        valid_sampler = SubsetRandomSampler(valid_idx)

        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=valid_sampler, 
            pin_memory=True, num_workers=num_workers)
    logging.info('### Train size: {}, Validation size: {} ###'.format(len(train_idx), len(valid_idx)))

    return train_loader, valid_loader


def get_test_loader(args):
    test_data = None
    if args.dataset == "mnist":
        test_transform = []
        test_transform += [transforms.ToTensor(),
                          transforms.Normalize(MNIST_mean, MNIST_std)]

        test_transform = transforms.Compose(test_transform)

        test_data = datasets.MNIST(root=args.data, train=False,
                                   download=True, transform=test_transform)
    elif args.dataset == "cifar":
        test_transform = []

        test_transform += [transforms.ToTensor(),
                          transforms.Normalize(CIFAR10_mean, CIFAR10_std)]

        test_transform = transforms.Compose(test_transform)

        test_data = datasets.CIFAR10(root=args.data, train=False,
                                   download=True, transform=test_transform)
    elif args.dataset == "svhn":
        test_transform = []
        test_transform += [transforms.ToTensor(),
                           transforms.Normalize(SVHN_mean, SVHN_std)]

        test_transform = transforms.Compose(test_transform)

        test_data = datasets.SVHN(root=args.data, split='test',
                                  download=True, transform=test_transform)
    elif "random" in args.dataset:
        norm = None
        if 'mnist' in args.dataset:
            norm = transforms.Normalize(MNIST_mean, MNIST_std)
        elif 'cifar' in args.dataset:
            norm = transforms.Normalize(CIFAR10_mean, CIFAR10_std)
        elif 'svhn' in args.dataset:
            norm = transforms.Normalize(SVHN_mean, SVHN_std)
        else:
            raise NotImplementedError("Other datasets not implemented")
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            norm
        ])        
        test_data = datasets.FakeData(size=10000, image_size=args.input_size[1:], num_classes=10, random_offset=args.seed,transform=test_transform)
    else:
        raise NotImplementedError("Other datasets not implemented")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=0)
    logging.info('### Test size: {} ###'.format(len(test_loader.dataset)))
    return test_loader
