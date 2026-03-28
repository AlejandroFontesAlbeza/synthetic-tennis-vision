import torch
from torch.utils.data import DataLoader, TensorDataset
from training.metrics import calculate_IoU
from unet.unet import Unet
from unet.custom_dataset import CustomDataset
from training.metrics import epoch_trained
import config_training

print("Running Training Tests...")
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
	img_dir = config_training.TEST_DATASET_IMAGES_TRAIN_DIR
	mask_dir = config_training.TEST_DATASET_MASKS_TRAIN_DIR
	num_classes = 10
	dataset = CustomDataset(img_dir, mask_dir)
	mask = dataset[0][1]
	assert mask.max() < num_classes
	assert mask.min() >= 0

def test_model_epoch_trained():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    in_channels = 3
    lr = 0.001

    # synthetic tensors dataset
    x_train = torch.randn(10, in_channels, 256, 256)
    y_train = torch.randint(0, num_classes, (10, 256, 256))
    x_val = torch.randn(5, in_channels, 256, 256)
    y_val = torch.randint(0, num_classes, (5, 256, 256))


    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = Unet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    val_loss = []

    for epoch in range(5):
        train_average_loss, val_average_loss, _ = epoch_trained(model, num_classes=num_classes, ignore_index=0,
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
