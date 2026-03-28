import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize_scalar


def homography(img_intersections, real_world_points):
    number_intersections = sorted(set(img_intersections.keys() & real_world_points.keys()))

    img_pts = np.array([img_intersections[i] for i in number_intersections])
    real_pts = np.array([real_world_points[i] for i in number_intersections])

    if len(number_intersections) >= 4:
        H, _ = cv2.findHomography(real_pts, img_pts)
        return H
    else:
        print("Not enough intersections for homography estimation.")
        return None

def camera_pose_estimation(H, cx, cy, f_prev = None):

    """
    Ideal f estimation based on the homography matrix
    """

    H = H / np.linalg.norm(H[:,0])

    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]

    def error(f):
        K = np.array([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        r1 = K_inv @ h1
        r2 = K_inv @ h2
        ortho = np.dot(r1, r2)
        norm_diff = np.linalg.norm(r1) - np.linalg.norm(r2)
        return ortho**2 + 0.1 * norm_diff**2

    if f_prev is not None:
        f_min = f_prev * 0.8
        f_prev = f_prev * 1.2
    else:
        f_min = 300
        f_prev = 5000

    res = minimize_scalar(error, bounds=(f_min, f_prev), method='bounded')
    f = res.x # x is the value we need from minimize_scalar Object

    """
    Camera pose estimation from homography
    """

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)



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

    C = -R.T @ t
    cam_position = C.flatten()

    R_wc = R.T
    rot = Rot.from_matrix(R_wc)
    rx,ry,rz = rot.as_euler('xyz', degrees = True)
    cam_rotation = np.round(np.array([rx, ry, rz]), 2)

    FOV_x = 2 * np.arctan((cx) / f)
    FOV_x_deg = np.rad2deg(FOV_x)

    return cam_position, cam_rotation, f, FOV_x_deg
