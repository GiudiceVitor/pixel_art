import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            folder_names (list): List of folder names to include in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.jpg') or img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, folder_names, transform=None, name1='real', name2='pixel-art'):
        """
        Args:
            root_dir (string): Directory with all the images.
            folder_names (list): List of folder names to include in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths_1 = []
        self.image_paths_2 = []

        for folder_name in folder_names:
            folder_path_1 = os.path.join(root_dir, name1, folder_name)
            folder_path_2 = os.path.join(root_dir, name2, folder_name)
            self.image_paths_1 += [os.path.join(folder_path_1, img) for img in os.listdir(folder_path_1) if img.endswith('.jpg')]
            self.image_paths_2 += [os.path.join(folder_path_2, img) for img in os.listdir(folder_path_2) if img.endswith('.jpg')]

    def __len__(self):
        return min(len(self.image_paths_1), len(self.image_paths_2))

    def __getitem__(self, idx):
        img_path_1 = self.image_paths_1[idx]
        image_1 = Image.open(img_path_1).convert('RGB')  # Convert to RGB in case some images are grayscale

        if self.transform:
            image_1 = self.transform(image_1)

        img_path_2 = self.image_paths_2[idx]
        image_2 = Image.open(img_path_2).convert('RGB')  # Convert to RGB in case some images are grayscale

        if self.transform:
            image_2 = self.transform(image_2)   

        return image_1, image_2
    
    
class CustomImageDataset2(Dataset):
    def __init__(self, root_dir, transform=None, name1='real', name2='pixel-art'):
        """
        Args:
            root_dir (string): Directory with all the images.
            folder_names (list): List of folder names to include in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths_1 = []
        self.image_paths_2 = []

        folder_path_1 = os.path.join(root_dir, name1)
        folder_path_2 = os.path.join(root_dir, name2)
        self.image_paths_1 = [os.path.join(folder_path_1, img) for img in os.listdir(folder_path_1) if img.endswith('.png')]
        self.image_paths_2 = [os.path.join(folder_path_2, img) for img in os.listdir(folder_path_2) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths_2)

    def __getitem__(self, idx):
        img_path_1 = self.image_paths_1[idx]
        image_1 = Image.open(img_path_1).convert('RGB')  # Convert to RGB in case some images are grayscale

        if self.transform:
            image_1 = self.transform(image_1)

        img_path_2 = self.image_paths_2[idx]
        image_2 = Image.open(img_path_2).convert('RGB')  # Convert to RGB in case some images are grayscale

        if self.transform:
            image_2 = self.transform(image_2)

        return image_1, image_2