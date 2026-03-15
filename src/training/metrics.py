import torch
from tqdm import tqdm


def calculate_IoU(preds, ground_truth, num_classes, ignore_index = None):

    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        preds_cls = (preds == cls)
        ground_truth_cls = (ground_truth == cls)

        intersection = (preds_cls & ground_truth_cls).sum().item()
        union = (preds_cls | ground_truth_cls).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    miou = torch.tensor(ious).nanmean().item()

    return ious, miou

def epoch_trained(model, num_classes, ignore_index, train_loader, val_loader, criterion, optimizer, device):

    """ Training loop for one epoch. """

    model.train()
    train_loss = 0.0
    print('training...')

    for idx, (images, masks) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_average_loss = train_loss / (idx + 1)

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    val_average_loss = val_loss / (idx + 1)
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    iou, miou = calculate_IoU(all_preds, all_masks, num_classes=num_classes, ignore_index=ignore_index)

    return train_average_loss, val_average_loss, miou
