from email import message

import cv2
import numpy as np
import argparse
import time

from inference.utils import get_device, get_model, get_tensor_transform, save_frame_info, send_udp_message
from inference.utils import VideoInference, draw_lines_and_intersections, write_gif_inference, draw_stats
from utils.dict_utils import inference_color_palette, intersections_lines, real_world_points
import config_inference
from inference.camera_pose import homography, camera_pose_estimation


def main():
    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--save_data', action='store_true', help='Save frame info')
    parser.add_argument('--save_video', action='store_true', help='Save output video as GIF')
    parser.add_argument('--show_mask', action='store_true', help='Show predicted mask')
    parser.add_argument('--show_stats', action='store_true', help='Show inference stats on display')
    parser.add_argument('--unique_color', action='store_false', help='Use a unique color for all lines')

    args = parser.parse_args()

    save_data = args.save_data
    save_video = args.save_video
    show_mask = args.show_mask
    show_stats = args.show_stats
    unique_color = args.unique_color
    print(f"save_data: {save_data}")
    print(f"save_video: {save_video}")
    print(f"show_mask: {show_mask}")
    print(f"show_stats: {show_stats}")
    print(f"unique_color: {unique_color}")

    device = get_device()
    print(f"Using device: {device}")

    model = get_model(config_inference.INFERENCE_MODEL_PATH, device)
    print("Model loaded successfully.")
    model.eval()

    transform = get_tensor_transform()

    video_inference = VideoInference(config_inference.INFERENCE_VIDEO_PATH, model,
                                    transform, device, inference_palette=inference_color_palette)
    width, height = video_inference.get_size()
    cx, cy = width / 2, height / 2
    f_prev = None
    resize = (width//2, 900)
    resize_no_mask = (width//2, height//2)

    frame_number = 0
    frames_list = []
    for frame, predicted_mask_resized, predicted_mask_color, inference_time, fps in video_inference:
        frame_with_lines, img_intersections = draw_lines_and_intersections(
            frame.copy(), predicted_mask_resized, inference_color_palette, intersections_lines,
            unique_color=unique_color
        )

        H = homography(img_intersections, real_world_points)

        if H is not None:
            cam_position, cam_rotation, f, FOV_x_deg = camera_pose_estimation(H, cx, cy, f_prev)
            f_prev = f
        else:
            print("Skipping Pose estimation due to insufficient intersections.")
            cam_position, cam_rotation, f, FOV_x_deg = [0,0,0], [0,0,0], 0, 0

        frame_with_mask = np.vstack((frame_with_lines, predicted_mask_color))
        if save_data:
            save_frame_info(config_inference.OUTPUT_DIR, frame_number, frame_with_mask, cam_position, cam_rotation, FOV_x_deg)
        else:
            pass
        if show_mask:
            display = frame_with_mask
            display_resized = cv2.resize(display, resize)
        else:
            display = frame_with_lines
            display_resized = cv2.resize(display, resize_no_mask)

        if show_stats:
            draw_stats(display_resized, inference_time, fps, cam_position, cam_rotation, FOV_x_deg)

        message = {
            "frame": int(frame_number),
            "camera_position": cam_position.tolist(),
            "camera_rotation": cam_rotation.tolist(),
            "FOV": float(FOV_x_deg),
            "timestamp": float(time.time())
        }
        send_udp_message(config_inference.IP, config_inference.PORT, message)

        frames_list.append(display_resized)
        cv2.imshow('Inference', display_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    if save_video:
        write_gif_inference(config_inference.OUTPUT_DIR, config_inference.OUTPUT_GIF_NAME, frames_list, fps=30)

    video_inference.release()
    cv2.destroyAllWindows()
    send_udp_message(config_inference.IP, config_inference.PORT, "exit")


if __name__ == "__main__":
    main()
