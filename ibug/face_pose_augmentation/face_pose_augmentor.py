import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any, Union, List
from .fpa import __file__ as fpa_init_file
from .fpa.utils import precompute_conn_point, model_completion_bfm
from .fpa import generate_profile_faces


__all__ = ['FacePoseAugmentor']


class FacePoseAugmentor(object):
    def __init__(self, model_folder: Optional[str] = None) -> None:
        self.fpa_models = FacePoseAugmentor.load_fpa_models(
            model_folder if model_folder else os.path.join(os.path.dirname(fpa_init_file), 'face_models'))

    @staticmethod
    def load_fpa_models(model_folder: str) -> Dict:
        fpa_models = dict()

        m_fwh = loadmat(os.path.join(model_folder, 'Model_FWH.mat'))
        fpa_models['Model_FWH'] = {
            'mu': m_fwh['Model_FWH']['mu'][0, 0].reshape(-1),
            'w': m_fwh['Model_FWH']['w'][0, 0].reshape((m_fwh['Model_FWH']['mu'][0, 0].size,
                                                        m_fwh['Model_FWH']['sigma'][0, 0].size)),
            'sigma': m_fwh['Model_FWH']['sigma'][0, 0].reshape(-1)}

        m_model_completion = loadmat(os.path.join(model_folder, 'Model_Completion.mat'))
        fpa_models['Model_Completion'] = {
            'indf_c': m_model_completion['indf_c'].reshape(-1).astype(int) - 1,
            'indf_c2b': m_model_completion['indf_c2b'].reshape(-1).astype(int) - 1,
            'trif_stitch': m_model_completion['trif_stitch'].reshape((3, -1)).astype(int) - 1,
            'trif_backhead': m_model_completion['trif_backhead'].reshape((3, -1)).astype(int) - 1}

        m_face_tri = loadmat(os.path.join(model_folder, 'Model_face_tri.mat'))
        fpa_models['tri'] = m_face_tri['tri'].astype(int) - 1

        m_tri_mouth = loadmat(os.path.join(model_folder, 'Model_tri_mouth.mat'))
        fpa_models['tri_mouth'] = m_tri_mouth['tri_mouth'].astype(int) - 1
        fpa_models['tri_plus'] = np.hstack([fpa_models['tri'], fpa_models['tri_mouth']])
        fpa_models['layer_width'] = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

        m_face_contour_trimed = loadmat(os.path.join(model_folder, 'Model_face_contour_trimed.mat'))
        fpa_models['face_contour_ind'] = m_face_contour_trimed['face_contour'].astype(int)

        m_fullmod_contour = loadmat(os.path.join(model_folder, 'Model_fullmod_contour.mat'))
        m_keypoints = loadmat(os.path.join(model_folder, 'Model_keypoints.mat'))
        fpa_models['keypoints'] = m_keypoints['keypoints'].reshape(-1).astype(int) - 1
        fpa_models['keypointsfull_contour'] = m_fullmod_contour['keypointsfull_contour'].reshape(-1).astype(int) - 1
        fpa_models['parallelfull_contour'] = [item[0][0].reshape(-1).astype(int) - 1 for item in
                                              m_fullmod_contour['parallelfull_contour']]

        fpa_models['conn_point_info'] = precompute_conn_point(fpa_models['tri_plus'], fpa_models['Model_Completion'])

        fpa_models['vertexm_full'], _ = model_completion_bfm(
            m_fwh['vertex_noear_BFM'], fpa_models['Model_FWH'],
            fpa_models['Model_Completion'], fpa_models['conn_point_info'])

        # create new keypoint index and its corresponding contours
        n_points = np.max(fpa_models['tri']) + 1
        keypoints_index = np.where(fpa_models['keypointsfull_contour'] < n_points)[0]
        fpa_models['keypoints_contour'] = fpa_models['keypointsfull_contour'][keypoints_index]
        fpa_models['parallel_contour'] = [fpa_models['parallelfull_contour'][i] for i in keypoints_index]

        return fpa_models

    def __call__(self, image: np.ndarray, tddfa_result: Dict, delta_poses: np.ndarray,
                 landmarks: Optional[np.ndarray] = None, warp_image: bool = True,
                 interpolation: Optional[int] = cv2.INTER_LINEAR,
                 border_mode: Optional[int] = cv2.BORDER_CONSTANT,
                 border_value: Any = 0) -> Union[Dict, List[Dict]]:
        if delta_poses.size > 0:
            if delta_poses.ndim > 1:
                results = []
                for corr_map, lms in zip(*generate_profile_faces(
                        delta_poses, tddfa_result, image, self.fpa_models, True,
                        landmarks=(landmarks + 1 if landmarks is not None else None))):
                    result = dict()
                    result['correspondence_map'] = corr_map
                    lms = np.ascontiguousarray(lms.transpose((0, 2, 1)))
                    lms[..., :2] -= 1
                    result['warped_landmarks'] = {'3d_style': lms[0], '2d_style': lms[1]}
                    if lms.shape[0] > 2:
                        result['warped_landmarks']['exact'] = lms[2]
                    if warp_image:
                        result['warped_image'] = cv2.remap(image, corr_map[..., 0].astype(np.float32),
                                                           corr_map[..., 1].astype(np.float32),
                                                           interpolation, borderMode=border_mode,
                                                           borderValue=border_value)
                    results.append(result)
                return results
            else:
                return self.__call__(image, tddfa_result, delta_poses[np.newaxis, ...], landmarks)[0]
        else:
            return []
