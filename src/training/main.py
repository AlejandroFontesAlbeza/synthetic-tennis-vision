from training.metrics import epoch_trained
from training.utils import device_selection, get_img_transform, get_data_loaders, get_model, optimizations



def train(train_img_path, valid_img_path, train_mask_path, valid_mask_path,
        num_classes, lr, batch_size, num_epochs, finetuning=False, model_path=None,
        new_model_path=None, step_lr=True, step_size=20):

    print('Training parameters:')
    print(f'num_classes: {num_classes}')
    print(f'lr: {lr}')
    print(f'batch_size: {batch_size}')
    print(f'num_epochs: {num_epochs}')
    print(f'finetuning: {finetuning}')
    print(f'model_path: {model_path}')
    print(f'new_model_path: {new_model_path}')
    print(f'step_lr: {step_lr}')
    print(f'step_size: {step_size}')

    device = device_selection()

    img_transform = get_img_transform()

    train_loader, val_loader = get_data_loaders(train_img_path, train_mask_path,
                                                valid_img_path, valid_mask_path,
                                                img_transform, batch_size)

    model = get_model(num_classes, device, finetuning, model_path)
    optimizer, scheduler, criterion = optimizations(model, lr, step_size)

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

    print('model saved')
    return model, new_model_path



