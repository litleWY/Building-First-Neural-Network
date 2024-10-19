# dataset.py - 数据集加载和转换
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# 确保数据集路径存在
os.makedirs('./datasets', exist_ok=True)

# 自定义数据集类
class CustomCIFAR10(Dataset):
    def __init__(self, train=True, transform=None):
        dataset_path = './datasets/cifar-10-batches-py'
        self.dataset = datasets.CIFAR10(root='./datasets', train=train, download=not os.path.exists(dataset_path))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据增强和加载
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 测试集只进行归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
