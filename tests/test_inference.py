from inference.utils import get_model, get_device, get_tensor_transform
from inference.utils import VideoInference
import config_inference
from utils.dict_utils import inference_color_palette
from inference.camera_pose import homography, camera_pose_estimation

print("Running Inference Tests...")

def test_homography_and_pose_estimation():
    ## Simulate Data
    img_intersections = {
        1: [100, 200],
        2: [200, 200],
        3: [200, 300],
        4: [100, 300]
    }

    real_world_points = {
        1: [0, 0],
        2: [1, 0],
        3: [1, 1],
        4: [0, 1]
    }

    H = homography(img_intersections, real_world_points)
    assert H is not None
    assert H.shape == (3, 3)
    cx, cy = 320, 240
    cam_position, cam_rotation, f, FOV_x_deg = camera_pose_estimation(H, cx, cy)
    assert cam_position is not None
    assert cam_rotation is not None
    assert f > 0
    assert FOV_x_deg > 0
