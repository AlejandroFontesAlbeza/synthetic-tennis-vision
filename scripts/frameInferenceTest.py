from camera_pose.utils.palette import inferenceColorPalette
from camera_pose.unet.training.architecture import Unet


import os
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


import torch
from torchvision import transforms



def inference(model_path, input_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img = Image.open(input_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    img_resize = np.array(img.resize((512, 512)))
    img_np = np.array(img_resize)
    
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_tensor)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        output = model(input_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        print(f'Inference time: {end-start}')

    prediced_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    predicted_mask_color = np.zeros((prediced_mask.shape[0], prediced_mask.shape[1], 3), dtype=np.uint8)
    for classIndex, color in inferenceColorPalette.items():
        predicted_mask_color[prediced_mask == classIndex] = color
    

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.subplot(1,2,2)
    plt.imshow(predicted_mask_color)
    plt.show()




if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    model_path = os.path.join(base_dir, "modelVersions/unet_modelV2.pth")
    input_path = os.path.join(base_dir, "src/camera_pose/datasetReal/train/images/frames_0030_png.rf.839bf6479edb3a1615413efa5220e2b2.jpg")
    inference(model_path, input_path=input_path)