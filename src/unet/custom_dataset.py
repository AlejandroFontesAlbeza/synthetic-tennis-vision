import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform = None):
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.masks_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask