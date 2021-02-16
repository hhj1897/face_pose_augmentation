from scipy.interpolate import interp1d
from .fpa import generate_profile_faces, retrieve_contour_landmark_aug


__all__ = ['FacePoseAugmentor']


class FacePoseAugmentor(object):
    def __init__(self) -> None:
        pass

    def __call__(self, image, tddfa_result, delta_poses, landmarks=None):
        pass
