import argparse
import os

import torch
from training.main import train



def main():

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_img_path = os.path.join(script_dir, "..", "data", "dataset", "train", "images")
    train_mask_path = os.path.join(script_dir, "..", "data", "dataset", "train", "masks")
    valid_img_path = os.path.join(script_dir, "..", "data", "dataset", "valid", "images")
    valid_mask_path = os.path.join(script_dir, "..", "data", "dataset", "valid", "masks")

    model, new_model_path = train(train_img_path, valid_img_path,
        train_mask_path, valid_mask_path,
        num_classes=args.num_classes, lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs,
        finetuning=args.finetuning, model_path=args.model_path, new_model_path=args.new_model_path, step_lr=args.step_lr,step_size=args.step_size)

    torch.save(model.state_dict(), new_model_path)


if __name__ == "__main__":
    main()
