import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from unet.custom_dataset import CustomDataset
from unet.unet import Unet

def device_selection():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Training on:', device)
    return device


def get_img_transform():
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.25),
        transforms.RandomGrayscale(p=0.25),
        transforms.ToTensor(),
    ])
    return img_transform

def get_data_loaders(train_img_path, train_mask_path, valid_img_path, valid_mask_path, img_transform, batch_size):
    train_dataset = CustomDataset(train_img_path, train_mask_path, img_transform)
    val_dataset = CustomDataset(valid_img_path, valid_mask_path, img_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()//2)

    return train_loader, val_loader

def get_model(num_classes, device, finetuning=False, model_path=None):
    model = Unet(in_channels=3, num_classes=num_classes).to(device)
    if finetuning and model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Finetuning, loading model path: {model_path}')
    else:
        print('Is not finetuning, training from scratch')
    return model

def optimizations(model, lr, step_size):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.6)
    return optimizer, scheduler, criterion

