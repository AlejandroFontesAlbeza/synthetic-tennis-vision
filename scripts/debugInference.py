from utils.palette import inferenceColorPalette
from unet.unet import Unet


import os
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as Rot


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
    img_np = np.array(img)
    width, height = img.size
    print(f"Input image size: {img.size}")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_tensor)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        output = model(input_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        print(f'Inference time: {end-start}')

    predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    predicted_mask_color = np.zeros((height, width, 3), dtype=np.uint8)


    lines = {}
    intersections_lines = {
        1: (1,6),
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
        20: (5,9)
    }
    intersections = {}

    real_intersections = {
        1: (-5.485, -11.885),
        2: (-5.485, -6.4),
        3: (-5.485, 6.4),
        4: (-5.485, 11.885),
        5: (-4.115, -11.885),
        6: (-4.115, -6.4),
        7: (-4.115, 6.4),
        8: (-4.115, 11.885),
        9: (0.0, -11.885),
        10: (0.0, -6.4),
        11: (0.0, 6.4),
        12: (0.0, 11.885),
        13: (4.115, -11.885),
        14: (4.115, -6.4),
        15: (4.115, 6.4),
        16: (4.115, 11.885),
        17: (5.485, -11.885),
        18: (5.485, -6.4),
        19: (5.485, 6.4),
        20: (5.485, 11.885)
    }


    for classIndex, color in inferenceColorPalette.items():
        mask = (predicted_mask_resized == classIndex)
        predicted_mask_color[mask] = color

    for classIndex, color in inferenceColorPalette.items():
        mask_binary = (predicted_mask_resized == classIndex).astype(np.uint8)

        if np.count_nonzero(mask_binary) < 100:  # Evita clases pequeñas
            continue
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)

        [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
        vx,vy,x,y = vx.item(), vy.item(), x.item(), y.item() # Convert to Python scalars
        lines[classIndex] = (vx, vy, x, y)

        cv2.line(img_np, (int(x - vx*1000), int(y - vy*1000)), (int(x + vx*1000), int(y + vy*1000)), color, 4)        
    
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
            intersections[index] = (x_int, y_int)

            cv2.circle(img_np, (int(x_int), int(y_int)), 8, (255,255,255), -1)
            cv2.putText(img_np, f'{index}', (int(x_int)+10, int(y_int)-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    

    number_intersecctions = sorted(set(intersections.keys() & real_intersections.keys()))

    img_pts = np.array([intersections[i] for i in number_intersecctions])
    real_pts = np.array([real_intersections[i] for i in number_intersecctions])

    if len(number_intersecctions) >= 4:
        H, _ = cv2.findHomography(real_pts, img_pts)
        print("Homography matrix:\n", H)

    

    FOV_x_deg = 51.282  # FOV horizontal
    FOV_x = np.deg2rad(FOV_x_deg)
    fx = (width / 2) / np.tan(FOV_x / 2)

    FOV_y = 2 * np.arctan((height / width) * np.tan(FOV_x / 2))
    fy = (height / 2) / np.tan(FOV_y / 2)

    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]], dtype=np.float32)
    

    H = H / np.linalg.norm(H[:,0])
    K_inv = np.linalg.inv(K)

    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]

    r1 = K_inv @ h1
    r2 = K_inv @ h2
    t = K_inv @ h3

    L = 1 / np.linalg.norm(r1)
    r1 *= L
    r2 *= L
    t *= L

    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    R_wc = R.T
    C = -R.T @ t
    camPosition = C.flatten()
    print(np.round(camPosition,2))

    rot = Rot.from_matrix(R_wc)
    rx,ry,rz = rot.as_euler('xyz', degrees = True)
    rx, ry, rz = np.round([rx, ry, rz], 2)
    print(rx, ry, rz)



    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.subplot(1,2,2)
    plt.imshow(predicted_mask_color)
    plt.show()




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", "unet_modelV0.pth")
    input_path = os.path.join(script_dir, "..", "data", "tennisMatch", "frames", "frame0249.png")
    inference(model_path, input_path=input_path)