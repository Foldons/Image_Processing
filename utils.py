import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets, transforms
def prep_dataLoadert(filename, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    ds = load_dataset("blanchon/UC_Merced")

    class UCMercedDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.data = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]['image']
            label = self.data[idx]['label']  # not really used for autoencoder
            if self.transform:
                image = self.transform(image)
            return image, label

    # Split and wrap in DataLoaders
    full_dataset = UCMercedDataset(ds['train'], transform=transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_minst(batch_size):
    transform = transforms.ToTensor()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4242, 0.3671, 0.1749],
                             std=[0.1022, 0.0893, 0.0850]),
        transforms.ToTensor(),
    ])


    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def load_cifar10(batch_size):
    transform = transforms.ToTensor()
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #])

    train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_data  = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader



if "__main__" == __name__:
    #train, test = prep_dataLoadert("blanchon/UC_Merced", 1)
    train , test = load_cifar10(1)
    for x, y in test:
        print(x.size())
        mean = x.mean(dim=[0,2,3])
        ted = x.std(dim=[0,2,3])

        print(mean)
        print(ted)
        breakpoint()





