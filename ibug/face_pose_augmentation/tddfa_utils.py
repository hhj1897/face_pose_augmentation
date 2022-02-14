import numpy as np
from typing import Tuple
from math import cos, atan2, asin
from .tddfa.utils.params import param_mean, param_std, u, w_shp, w_exp, u_base, w_shp_base, w_exp_base


__all__ = ['matrix2angle', 'decompose_camera_matrix', 'parse_param', 'parse_param_pose', 'reconstruct_from_3dmm']


def matrix2angle(rot_mat: np.ndarray) -> Tuple[float, float, float]:
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        rot_mat: (3,3). rotation matrix
    Returns (assumes rotation in x-y-z order):
        x: yaw
        y: pitch
        z: roll
    """
    if -1.0 < rot_mat[2, 0] < 1.0:
        x = -asin(rot_mat[2, 0])    # It is minus in the reference
        y = atan2(rot_mat[2, 1] / cos(x), rot_mat[2, 2] / cos(x))
        z = atan2(rot_mat[1, 0] / cos(x), rot_mat[0, 0] / cos(x))
    else:
        # Gimbal lock
        z = 0   # can be anything
        if rot_mat[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(rot_mat[0, 1], rot_mat[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-rot_mat[0, 1], -rot_mat[0, 2])

    return x, y, z


def decompose_camera_matrix(cam_mat: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """ decompose camera matrix.
    Args:
        cam_mat: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        rot_mat: (3, 3). rotation matrix.
        t3d: (3,). 3d translation.
    """
    t3d = cam_mat[:, 3]
    s = (np.linalg.norm(cam_mat[0, :3]) + np.linalg.norm(cam_mat[1, :3])) / 2.0
    r1 = cam_mat[0, :3] / np.linalg.norm(cam_mat[0, :3])
    r2 = cam_mat[1, :3] / np.linalg.norm(cam_mat[1, :3])
    r3 = np.cross(r1, r2)
    rot_mat = np.vstack((r1, r2, r3))

    return s, rot_mat, t3d


def parse_param(param: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ works for both numpy and tensor """
    cam_mat = param[:12].reshape(3, -1)
    f_rot = cam_mat[:, :3]
    tr = cam_mat[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return f_rot, tr, alpha_shp, alpha_exp


def parse_param_pose(param: np.ndarray, pose_pref: int = 0) -> Tuple[float, float, float, np.ndarray, float]:
    param = param * param_std + param_mean
    cam_mat = param[:12].reshape(3, -1)
    s, rot_mat, t3d = decompose_camera_matrix(cam_mat)
    yaw, pitch, roll = matrix2angle(rot_mat)

    # Respond to pose_pref:
    # pose_pref == 1: limit pitch to the range of -90.0 ~ 90.0
    # pose_pref == 2: limit yaw to the range of -90.0 ~ 90.0 (already satisfied)
    # pose_pref == 3: limit roll to the range of -90.0 ~ 90.0
    # otherwise: minimise total rotation, min(abs(pitch) + abs(yaw) + abs(roll))
    if pose_pref != 2:
        alt_pitch = pitch - np.pi if pitch > 0.0 else pitch + np.pi
        alt_yaw = -np.pi - yaw if yaw < 0.0 else np.pi - yaw
        alt_roll = roll - np.pi if roll > 0.0 else roll + np.pi
        if (pose_pref == 1 and -np.pi / 2.0 < alt_pitch < np.pi / 2.0 or
                pose_pref == 3 and -np.pi / 2.0 < alt_roll < np.pi / 2.0 or
                pose_pref not in (1, 2, 3) and
                abs(alt_pitch) + abs(alt_yaw) + abs(alt_roll) < abs(pitch) + abs(yaw) + abs(roll)):
            pitch, yaw, roll = alt_pitch, alt_yaw, alt_roll

    return pitch, yaw, roll, t3d, s


def reconstruct_from_3dmm(param: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    param = param * param_std + param_mean
    f_rot, tr, alpha_shp, alpha_exp = parse_param(param)
    vertex = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
    pts68 = (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F')
    return vertex, pts68, f_rot, tr
