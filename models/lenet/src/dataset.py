import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

class BreastTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, val_ratio=0.2, seed=42):
        self.image_paths = []
        self.labels = []

        for patient_id in os.listdir(root_dir):
            patient_folder = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_folder):
                continue

            for class_label in ['0', '1']:
                class_folder = os.path.join(patient_folder, class_label)
                if not os.path.exists(class_folder):
                    continue
                for img_name in os.listdir(class_folder):
                    if img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(class_folder, img_name))
                        self.labels.append(int(class_label))

        # Split the data into train and validation sets
        random.seed(seed)
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        split = int(len(indices) * (1 - val_ratio))
        
        if train:
            indices = indices[:split]
        else:
            indices = indices[split:]
            
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        self.transform = transform or transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(image), label

def get_dataloaders(data_dir, batch_size=32, val_ratio=0.2, seed=42):
    """
    Create training and validation dataloaders
    
    Args:
        data_dir: Path to the dataset
        batch_size: Batch size for training and validation
        val_ratio: Ratio of validation data
        seed: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ])
    
    train_dataset = BreastTumorDataset(
        root_dir=data_dir,
        transform=transform,
        train=True,
        val_ratio=val_ratio,
        seed=seed
    )
    
    val_dataset = BreastTumorDataset(
        root_dir=data_dir,
        transform=transform,
        train=False,
        val_ratio=val_ratio,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader 