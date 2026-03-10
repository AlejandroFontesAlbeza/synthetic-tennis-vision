import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.unet.architecture import CustomDataset, Unet
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser(description='Train a UNet model for image segmentation')

parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for segmentation')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--finetuning', action='store_true', help='Whether to perform fine-tuning')
parser.add_argument('--last_model_path', type=str, default='models/unet_modelV2.pth', help='Path to the pre-trained model for fine-tuning')
parser.add_argument('--new_model_path', type=str, default='models/unet_modelV3.pth', help='Path to save the fine-tuned/trained model')
parser.add_argument('--step_lr', action='store_true', help='Whether to use StepLR')

args = parser.parse_args()

def main(train_img_path, valid_img_path, train_mask_path, valid_mask_path,
        num_classes, lr, batch_size, num_epochs, finetuning=False, last_model_path=None, new_model_path=None, step_lr=True):

    print('Starting training with the following parameters:')
    print(f'  - Number of classes: {num_classes}')
    print(f'  - Learning rate: {lr}')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Number of epochs: {num_epochs}')
    print(f'  - Use StepLR: {step_lr}')
    print(f'  - Finetuning: {finetuning}')
    if finetuning:
        print(f'  - Last model path: {last_model_path}')
        print(f'  - New model path: {new_model_path}')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training on:', device)

    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.25),
        transforms.RandomGrayscale(p=0.25),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(train_img_path, train_mask_path, img_transform)
    val_dataset = CustomDataset(valid_img_path, valid_mask_path, img_transform)

    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()//2)


    model = Unet(in_channels=3, num_classes=num_classes).to(device)
    if finetuning == False:
        print('Is not finetuning, training from scratch')
        None
    else:
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f'Finetuning, loading model path: {last_model_path}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print('training...')

        for index, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            masks = masks.to(device)

            
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_average_loss = train_loss / (index + 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, (images, masks) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_average_loss = val_loss / (index + 1)
        if step_lr == True:
            scheduler.step()
        else:
            print('Not using StepLR, lr is fixed')
        
        print(f'Epoch: {epoch} / {num_epochs}, train_loss: {train_average_loss}, val_loss: {val_average_loss}, lr: {optimizer.param_groups[0]["lr"]}') 

    torch.save(model.state_dict(), new_model_path)
    print('model saved') 



if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    train_img_path = os.path.join(base_dir, "data/dataset/train/images")
    train_mask_path = os.path.join(base_dir, "data/dataset/train/masks")
    valid_img_path = os.path.join(base_dir, "data/dataset/valid/images")
    valid_mask_path = os.path.join(base_dir, "data/dataset/valid/masks")


    main(train_img_path, valid_img_path,
        train_mask_path, valid_mask_path,
        num_classes=args.num_classes, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs,
        finetuning=args.finetuning, last_model_path=args.last_model_path, new_model_path=args.new_model_path, step_lr=args.step_lr)