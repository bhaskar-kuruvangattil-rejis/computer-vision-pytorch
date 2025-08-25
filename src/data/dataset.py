import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import pandas as pd
from typing import Optional, Callable, Tuple, Any


class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images and labels.
    """
    
    def __init__(self, 
                 csv_file: str = None, 
                 root_dir: str = None, 
                 image_paths: list = None,
                 labels: list = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            csv_file (str): Path to CSV file with image paths and labels
            root_dir (str): Directory with all images
            image_paths (list): List of image paths (alternative to CSV)
            labels (list): List of labels (alternative to CSV)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        
        if csv_file is not None:
            self.data_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
        elif image_paths is not None and labels is not None:
            self.data_frame = pd.DataFrame({
                'image_path': image_paths,
                'label': labels
            })
            self.root_dir = root_dir
        else:
            raise ValueError("Either csv_file or (image_paths, labels) must be provided")
        
        self.transform = transform
    
    def __len__(self):
        """TODO: Add description for __len__"""
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size: Tuple[int, int] = (224, 224), 
                  augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms.
    
    Args:
        input_size (tuple): Input size for the model (height, width)
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # Basic transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if augment:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform
    
    return train_transform, val_transform


def get_dataloaders(dataset_name: str = 'CIFAR10',
                   data_dir: str = './data/raw',
                   batch_size: int = 32,
                   num_workers: int = 4,
                   input_size: Tuple[int, int] = (224, 224)) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders for common datasets.
    
    Args:
        dataset_name (str): Name of the dataset ('CIFAR10', 'CIFAR100', 'MNIST', etc.)
        data_dir (str): Directory to download/store data
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        input_size (tuple): Input size for the model
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    train_transform, val_transform = get_transforms(input_size, augment=True)
    
    if dataset_name.upper() == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                       download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                     download=True, transform=val_transform)
    
    elif dataset_name.upper() == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, 
                                        download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, 
                                      download=True, transform=val_transform)
    
    elif dataset_name.upper() == 'MNIST':
        # Convert MNIST to RGB for consistency
        mnist_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.MNIST(root=data_dir, train=True, 
                                     download=True, transform=mnist_transform)
        val_dataset = datasets.MNIST(root=data_dir, train=False, 
                                   download=True, transform=mnist_transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                        f"Supported: CIFAR10, CIFAR100, MNIST")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def create_custom_dataloader(csv_file: str = None,
                           root_dir: str = None,
                           image_paths: list = None,
                           labels: list = None,
                           batch_size: int = 32,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           input_size: Tuple[int, int] = (224, 224),
                           augment: bool = True) -> DataLoader:
    """
    Create a custom data loader.
    
    Args:
        csv_file (str): Path to CSV file with image paths and labels
        root_dir (str): Directory with all images
        image_paths (list): List of image paths
        labels (list): List of labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        input_size (tuple): Input size for the model
        augment (bool): Whether to apply data augmentation
    
    Returns:
        DataLoader: Custom data loader
    """
    
    transform = get_transforms(input_size, augment)[0 if augment else 1]
    
    dataset = CustomImageDataset(csv_file=csv_file, 
                               root_dir=root_dir,
                               image_paths=image_paths,
                               labels=labels,
                               transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers)
