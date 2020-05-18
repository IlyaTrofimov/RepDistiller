from __future__ import print_function

import os
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

MEAN = [0.50707516, 0.48654887, 0.44091784]
STD = [0.26733429, 0.25643846, 0.27615047]

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CIFAR100BackCompat(datasets.CIFAR100):
     """
     CIFAR100Instance+Sample Dataset
     """

     #@property
     #def train_labels(self):
     #    return self.targets

     #@property
     #def test_labels(self):
     #    return self.targets

     #@property
     #def train_data(self):
     #    return self.data

     #@property
     #def test_data(self):
     #    return self.data

class CIFAR100Instance(CIFAR100BackCompat):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=4, is_instance=False, part = 1):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    n_train = 50000
    split = 40000
    indices = list(range(n_train))

    n_data = split//part

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)

        val_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=test_transform)

        val_set = torch.utils.data.Subset(val_set, indices[split:])
    else:
        raise NotImplementedError('NOT IMPLEMENTED')

    #
    #
    #
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_data])

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)

    val_loader = DataLoader(val_set,
                              batch_size=batch_size,
                              shuffle = False,
                              num_workers=num_workers)

    #
    #
    #
    test_set = CIFAR100Instance(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    print('NORMAL DATA LOADER')

    if is_instance:
        return train_loader, val_loader, test_loader, n_data
    else:
        return train_loader, val_loader, test_loader


class CIFAR100InstanceSample(CIFAR100BackCompat):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0, count = 0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.train_data)
            label = self.train_labels
        else:
            #num_samples = len(self.test_data)
            #label = self.test_labels
            raise NotImplementedError('NOT USED FOR TEST')

        self.cls_positive = [[] for i in range(num_classes)]

        num_samples = count # take count object from the begining

        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0, part = 1):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    n_train = 50000
    split = 40000
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split//part])

    n_train_data = split//part

    #
    #
    #
    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent, count = n_train_data)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)

    #
    #
    #
    val_set = CIFAR100Instance(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=test_transform)

    val_set = torch.utils.data.Subset(val_set, indices[split:])

    val_loader = DataLoader(val_set,
                              batch_size=batch_size,
                              shuffle = False,
                              num_workers=num_workers)
    #
    #
    #
    test_set = CIFAR100Instance(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, val_loader, n_train_data
