import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def tf(train=False):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if train:
        transformer.transforms.append(transforms.RandomHorizontalFlip())

    return transformer


def form_datasets():
    transform_train = tf(True)

    transform_test = tf(False)

    train_dataset = datasets.CIFAR10(
        root='..',
        train=True,
        transform=transform_train,
        download=False
    )

    test_dataset = datasets.CIFAR10(
        root='..',
        train=False,
        transform=transform_test,
        download=False
    )

    return train_dataset, test_dataset


def form_dataloader(dataset, batch_size, test=False, pin=False):
    shuffle = False if test else True
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=shuffle,
        pin_memory=pin,
    )
    return dataloader


if __name__ == '__main__':
    train_dataset, test_dataset = form_datasets()
    print(test_dataset)
    data_loader = form_dataloader(test_dataset, 64, test=True, pin=True)
    print(len(data_loader.dataset))


