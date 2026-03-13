import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from unet.custom_dataset import CustomDataset
from unet.unet import Unet
from training.metrics import epoch_trained
import os

import argparse

parser = argparse.ArgumentParser(description='Train a UNet model for image segmentation')

parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for segmentation')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--finetuning', action='store_true', help='Whether to perform fine-tuning')
parser.add_argument('--model_path', type=str, default='models/unet_modelV1.pth', help='Path to the pre-trained model for fine-tuning')
parser.add_argument('--new_model_path', type=str, default='models/unet_modelV3.pth', help='Path to save the fine-tuned/trained model')
parser.add_argument('--step_lr', action='store_true', help='Whether to use StepLR')
parser.add_argument('--step_size', type=int, default=20, help='Step size for StepLR')

args = parser.parse_args()

def main(train_img_path, valid_img_path, train_mask_path, valid_mask_path,
        num_classes, lr, batch_size, num_epochs, finetuning=False, model_path=None, new_model_path=None, step_lr=True, step_size=20):

    print('Starting training with the following parameters:')
    print(f'  - Number of classes: {num_classes}')
    print(f'  - Learning rate: {lr}')
    print(f'  - Batch size: {batch_size}')
    print(f'  - Number of epochs: {num_epochs}')
    print(f'  - Use StepLR: {step_lr}')
    print(f'  - Finetuning: {finetuning}')
    print(f'  - Step size: {step_size}')
    print(f'  - Model path: {model_path}')
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
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Finetuning, loading model path: {model_path}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.6)

    for epoch in range(num_epochs):
        
        train_average_loss, val_average_loss, miou = epoch_trained(model, num_classes, ignore_index=0,
                                                            train_loader=train_loader,
                                                            val_loader=val_loader, criterion=criterion,
                                                            optimizer=optimizer, device=device)

        if step_lr == True:
            scheduler.step()
        else:
            print('Not using StepLR, lr is fixed')
        
        print(f'Epoch: {epoch} / {num_epochs}')
        print(f'Training Loss: {train_average_loss:.4f}')
        print(f'Validation Loss: {val_average_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'mIoU: {miou * 100:.2f}%')
        
    torch.save(model.state_dict(), new_model_path)
    print('model saved') 



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_img_path = os.path.join(script_dir, "..", "..", "data", "dataset", "train", "images")
    train_mask_path = os.path.join(script_dir, "..", "..", "data", "dataset", "train", "masks")
    valid_img_path = os.path.join(script_dir, "..", "..", "data", "dataset", "valid", "images")
    valid_mask_path = os.path.join(script_dir, "..", "..", "data", "dataset", "valid", "masks")


    main(train_img_path, valid_img_path,
        train_mask_path, valid_mask_path,
        num_classes=args.num_classes, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs,
        finetuning=args.finetuning, model_path=args.model_path, new_model_path=args.new_model_path, step_lr=args.step_lr,step_size=args.step_size)