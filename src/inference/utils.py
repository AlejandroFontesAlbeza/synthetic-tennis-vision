import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import json
import socket
from PIL import Image


import torch
from unet.unet import Unet
from torchvision import transforms

def draw_lines_and_intersections(img_np, predicted_mask_resized, inference_color_palette, intersections_lines, min_pixels=200, unique_color=True):
    lines = {}
    intersections = {}
    lines_distance = 2000

    # Si la imagen es BGR y la paleta es RGB, convierte los colores de la paleta a BGR
    def to_bgr(color):
        return (color[2], color[1], color[0])

    fixed_color = (0, 255, 0)

    # Dibuja líneas ajustadas a los contornos de cada clase
    for class_index, color in inference_color_palette.items():
        if class_index == 0:
            continue
        mask_binary = (predicted_mask_resized == class_index).astype(np.uint8)
        if np.count_nonzero(mask_binary) < min_pixels:
            continue
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = vx.item(), vy.item(), x.item(), y.item()
        lines[class_index] = (vx, vy, x, y)
        use_this_color = fixed_color if unique_color else to_bgr(color)
        cv2.line(img_np, (int(x - vx * lines_distance), int(y - vy * lines_distance)), (int(x + vx * lines_distance), int(y + vy * lines_distance)), use_this_color, 3)

    # Dibuja intersecciones
    for index, (c1, c2) in intersections_lines.items():
        if c1 in lines and c2 in lines:
            vx1, vy1, x1, y1 = lines[c1]
            vx2, vy2, x2, y2 = lines[c2]
            A = np.array([[vx1, -vx2], [vy1, -vy2]])
            b = np.array([x2 - x1, y2 - y1])
            t, s = np.linalg.solve(A, b)
            x_int = int(x2 + vx2 * s)
            y_int = int(y2 + vy2 * s)
            intersections[index] = (x_int, y_int)
            cv2.circle(img_np, (x_int, y_int), 8, (255, 255, 255), -1)
            cv2.putText(img_np, f'{index}', (x_int + 10, y_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    return img_np, intersections

def draw_stats(frame, inference_time, fps, cam_position, cam_rotation, FOV_x_deg):
    # Formateo de arrays con redondeo a 2 decimales (solo para display)
    cam_pos_str = ', '.join(f'{v:.2f}' for v in cam_position)
    cam_rot_str = ', '.join(f'{v:.2f}' for v in cam_rotation)


    text_inference = f'{inference_time:.2f} ms'
    text_fps = f'{fps:.1f} FPS'
    text_position  = f'Cam Position  : [{cam_pos_str}]'
    text_rotation  = f'Cam Rotation  : [{cam_rot_str}]'
    text_fov = f'FOV_H : {FOV_x_deg:.2f} degrees'

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.4
    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 1
    line_height = 20
    x = 10
    y_start = 20

    box_x1 = 0
    box_y1 = 0
    box_x2 = 305
    box_y2 = y_start + 3 * line_height + 40  # Ajusta según el número de líneas

    # Dibujar la caja semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)  # Caja negra
    alpha = 0.8  # Opacidad
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Renderizado en pantalla
    cv2.putText(frame, text_inference, (x, y_start), font, font_scale, red, thickness)
    cv2.putText(frame, text_fps, (x, y_start + line_height), font, font_scale, red, thickness)
    cv2.putText(frame, text_position,  (x, y_start + 2 * line_height), font, font_scale, green, thickness)
    cv2.putText(frame, text_rotation,  (x, y_start + 3 * line_height), font, font_scale, green, thickness)
    cv2.putText(frame, text_fov, (x, y_start + 4 * line_height), font, font_scale, green, thickness)

class VideoInference:
    def __init__(self, video_path, model, transform, device, inference_palette=None):
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_path = video_path
        self.model = model
        self.transform = transform
        self.device = device
        self.inference_palette = inference_palette

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(2):
                _ = self.model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            output = self.model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            inference_time = (end - start) * 1000
            fps = 1000 / inference_time if inference_time > 0 else float('inf')
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        predicted_mask_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for class_index, color in self.inference_palette.items():
            mask = (predicted_mask_resized == class_index)
            predicted_mask_color[mask] = color
        predicted_mask_color = cv2.cvtColor(predicted_mask_color, cv2.COLOR_RGB2BGR)
        return frame, predicted_mask_resized, predicted_mask_color, inference_time, fps

    def get_size(self):
        return self.width, self.height

    def release(self):
        self.cap.release()


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_model(model_path, device):
    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def get_tensor_transform():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform

def save_frame_info(output_dir, frame_number, frame_with_mask, cam_position, cam_rotation, FOV=90):
    frames_dir = os.path.join(output_dir, "frames")
    jsons_dir = os.path.join(output_dir, "jsons")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_path, frame_with_mask)

    pose_data = {
        "frame_number": frame_number,
        "camera_position": cam_position.tolist(),
        "camera_rotation": cam_rotation.tolist(),
        "FOV": FOV
    }
    pose_path = os.path.join(jsons_dir, f"pose_{frame_number:04d}.json")
    with open(pose_path, 'w') as f:
        json.dump(pose_data, f, indent=4)

    print(f"Saved frame to {frame_path} and pose info to {pose_path}")


def write_gif_inference(output_dir, gif_name, frames, fps=15):
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, gif_name)
    pil_frames = []
    for frame in tqdm(frames, desc="Generating GIF"):
        # Convertir BGR (OpenCV) → RGB
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)
        pil_frames.append(pil_image)
    # Guardar GIF
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0 )
    print(f"GIF saved to {gif_path}")


def send_udp_message(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(message).encode(), (ip, port))

