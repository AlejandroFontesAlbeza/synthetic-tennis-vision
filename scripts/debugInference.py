from src.utils.palette import inferenceColorPalette
from src.unet.architecture import Unet


import os
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2


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
    predicted_mask_lines = np.zeros((prediced_mask.shape[0], prediced_mask.shape[1], 3), dtype=np.uint8)
    lines = {}
    intersections_lines = {1: (1,6),
                           2: (1,7),
                           3: (1,8),
                           4: (1,9),
                           5: (2,6),
                           6: (2,7),
                           7: (2,8),
                           8: (2,9),
                           9: (3,6),
                           10: (3,7),
                           11: (3,8),
                           12: (3,9),
                           13: (4,6),
                           14: (4,7),
                           15: (4,8),
                           16: (4,9),
                           17: (5,6),
                           18: (5,7),
                           19: (5,8),
                           20: (5,9)}
    intersections = {}


    for classIndex, color in inferenceColorPalette.items():
        predicted_mask_color[prediced_mask == classIndex] = color
        contours,_ = cv2.findContours((prediced_mask == classIndex).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        min_contour_length = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_length]

        [vx,vy,x,y] = cv2.fitLine(valid_contours[0], cv2.DIST_L2,0,0.01,0.01)
        vx,vy,x,y = vx.item(), vy.item(), x.item(), y.item() # Convert to Python scalars
        lines[classIndex] = (vx, vy, x, y)

        cv2.line(predicted_mask_lines, (int(x - vx*1000), int(y - vy*1000)), (int(x + vx*1000), int(y + vy*1000)), color, 2)

    
    for index , (c1,c2) in intersections_lines.items():
        if c1 in lines and c2 in lines:
            vx1, vy1, x1, y1 = lines[c1]
            vx2, vy2, x2, y2 = lines[c2]
            A = np.array([[vx1, -vx2],[vy1, -vy2]])
            b = np.array([x2 - x1, y2 - y1])
            t, s = np.linalg.solve(A, b)

            """ x_int = int(x1 + vx1 * t)
            y_int = int(y1 + vy1 * t) 
            Or next: """
            x_int = int(x2 + vx2 * s)
            y_int = int(y2 + vy2 * s)
            intersections[(c1,c2)] = (x_int, y_int)

            cv2.circle(predicted_mask_lines, (int(x_int), int(y_int)), 5, (255,255,255), -1)
            cv2.putText(predicted_mask_lines, f'{index}', (int(x_int)+5, int(y_int)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    
    plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1)
    plt.imshow(img_np)
    plt.subplot(1,3,2)
    plt.imshow(predicted_mask_color)
    plt.subplot(1,3,3)
    plt.imshow(predicted_mask_lines)
    plt.show()




if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    model_path = os.path.join(base_dir, "models/unet_modelV2.pth")
    input_path = os.path.join(base_dir, "data/tennisMatch/output/frames2215.png")
    inference(model_path, input_path=input_path)