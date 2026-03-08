import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from camera_pose.unet.training.architecture import CustomDataset, Unet
from tqdm import tqdm
import os


def main(train_img_path, valid_img_path, train_mask_path, valid_mask_path, num_classes, batch_size, num_epochs, finetuning=False):

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    model = Unet(in_channels=3, num_classes=num_classes).to(device)
    if finetuning == False:
        print('Is not finetuning, training from scratch')
        None
    else:
        print('Finetuning, loading model path')
        model.load_state_dict(torch.load('modelVersions/unet_modelV1.pth', map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

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
        #scheduler.step()
        
        print(f'Epoch: {epoch} / {num_epochs}, train_loss: {train_average_loss}, val_loss: {val_average_loss}, lr: {optimizer.param_groups[0]["lr"]}') 

    torch.save(model.state_dict(), 'modelVersions/unet_modelV2.pth')
    print('model saved') 



if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    train_img_path = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/train/images")
    train_mask_path = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/train/masks")
    valid_img_path = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/valid/images")
    valid_mask_path = os.path.join(base_dir, "camera_pose/datasetStructuredSynthetic/valid/masks")

    main(train_img_path=train_img_path, valid_img_path=valid_img_path,
        train_mask_path=train_mask_path, valid_mask_path=valid_mask_path,
        num_classes=10, batch_size=2, num_epochs=20, finetuning=True)