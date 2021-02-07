import numpy as np
from math import cos, atan2, asin
from .tddfa.utils.params import param_mean, param_std, u, w_shp, w_exp, u_base, w_shp_base, w_exp_base


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    """

    if R[2, 0] != 1 or R[2, 0] != -1:
        # It is minus in the reference
        x = -asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


def parse_param_pose(param):
    param = param * param_std + param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    yaw, pitch, roll = matrix2angle(R)  # yaw, pitch, roll
    return yaw, pitch, roll, t3d, s


def reconstruct_from_3dmm(param, dense=True):
    param = param * param_std + param_mean
    fR, t3d, alpha_shp, alpha_exp = parse_param(param)
    vertex = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
    pts68 = (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F')
    return vertex, pts68, fR, t3d
