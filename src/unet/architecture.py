import torch
import torch.nn as nn
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
    

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        ### Encoder
        
        #block1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #block2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block3
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block4
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)



        ###Bottlneck

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )


        ###Decoder

        #Up1
        self.up1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024,512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512, kernel_size=3,padding=1),
            nn.ReLU()
        )


        #Up2
        self.up2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3,padding=1),
            nn.ReLU()
        )

        #Up3
        self.up3 = nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3,padding=1),
            nn.ReLU()
        )

        #Up4
        self.up4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3,padding=1),
            nn.ReLU()
        )


        ### Output Layer

        self.out = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x):

        ###Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        ###Bottleneck
        b = self.bottleneck(p4)
        
        ###Decoder

        #U1
        u1 = self.up1(b)
        u1 = torch.cat([u1,x4], dim=1)
        u1 = self.dec1(u1)

        #U2
        u2 = self.up2(u1)
        u2 = torch.cat([u2,x3], dim=1)
        u2 = self.dec2(u2)

        #U3
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.dec3(u3)

        #U4
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.dec4(u4)

        return self.out(u4)
    

