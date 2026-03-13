import torch
from torchvision import transforms
import os
from training.metrics import calculate_IoU
from unet.unet import Unet
from unet.custom_dataset import CustomDataset
from training.metrics import epoch_trained
from torch.utils.data import DataLoader

def test_calculate_iou_simple_case():
	preds = torch.tensor([[0, 1], [1, 1]])
	labels = torch.tensor([[0, 1], [0, 1]])
	ious, miou = calculate_IoU(preds, labels, num_classes=2)
	assert abs(miou - 0.5833) < 1e-4

def test_unet_forward():
	model = Unet(in_channels=3, num_classes=10)
	x = torch.randn(1, 3, 512, 512)
	y = model(x)
	assert y.shape == (1, 10, 512, 512)

def test_mask_class_range():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	img_dir = os.path.join(script_dir, "..","..", "data", "test_dataset", "train", "images")
	mask_dir = os.path.join(script_dir, "..","..", "data", "test_dataset", "train", "masks")
	num_classes = 10
	dataset = CustomDataset(img_dir, mask_dir)
	mask = dataset[0][1]
	assert mask.max() < num_classes
	assert mask.min() >= 0

def test_training_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_img_path = os.path.join(script_dir, "..", "..", "data", "test_dataset", "train", "images")
    train_mask_path = os.path.join(script_dir, "..", "..", "data", "test_dataset", "train", "masks")
    valid_img_path = os.path.join(script_dir, "..", "..", "data", "test_dataset", "valid", "images")
    valid_mask_path = os.path.join(script_dir, "..", "..", "data", "test_dataset", "valid", "masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    in_channels = 3
    ignore_index = 0
    lr = 0.001
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
		transforms.ToTensor()])

    train_dataset = CustomDataset(train_img_path, train_mask_path, img_transform=transform)
    val_dataset = CustomDataset(valid_img_path, valid_mask_path, img_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=os.cpu_count()//2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=os.cpu_count()//2)

    model = Unet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	
    train_loss = []
    val_loss = []
	
    for epoch in range(5):
        train_average_loss, val_average_loss, _ = epoch_trained(model, num_classes=num_classes, ignore_index=ignore_index,
																   train_loader=train_loader,
																   val_loader=val_loader, criterion=criterion,
																   optimizer=optimizer, device=device)
		
        train_loss.append(train_average_loss)
        val_loss.append(val_average_loss)
        print(f'Epoch: {epoch} / 5')
        print(f'Training Loss: {train_average_loss:.4f}')
        print(f'Validation Loss: {val_average_loss:.4f}')
		
    assert train_loss[-1] < train_loss[0] or abs(train_loss[-1] - train_loss[0]) < lr
    assert val_loss[-1] < val_loss[0] or abs(val_loss[-1] - val_loss[0]) < lr
	