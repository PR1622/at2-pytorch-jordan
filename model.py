# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 classes
CLASSES = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def get_transforms():
    # Normalize to [-1, 1] range around 0 with mean/std 0.5
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

def get_dataloaders(data_root: str, batch_size: int, num_workers: int = 2):
    """
    Builds train/test DataLoaders for CIFAR-10.
    data_root: path where CIFAR10 will be downloaded/stored.
    batch_size: per-process batch size.
    """
    tfm = get_transforms()

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=tfm
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tfm
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return trainloader, testloader

class Net(nn.Module):
    """
    Simple CNN for CIFAR-10 (from PyTorch tutorial):
    Conv(3->6,5) + ReLU + MaxPool
    Conv(6->16,5) + ReLU + MaxPool
    FC(16*5*5->120) + ReLU
    FC(120->84) + ReLU
    FC(84->10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
