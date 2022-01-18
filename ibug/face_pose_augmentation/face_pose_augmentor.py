import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any, Union, List, Sequence
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
                for idx, (corr_map, lms) in enumerate(zip(*generate_profile_faces(
                        delta_poses, tddfa_result, image, self.fpa_models, True,
                        landmarks=(landmarks + 1 if landmarks is not None else None)))):
                    result = dict()
                    result['correspondence_map'] = corr_map
                    lms = np.ascontiguousarray(lms.transpose((0, 2, 1)))
                    lms[..., :2] -= 1
                    result['warped_landmarks'] = {'3d_style': lms[0], '2d_style': lms[1]}
                    if lms.shape[0] > 2:
                        result['warped_landmarks']['projected_3d'] = lms[2]
                        result['warped_landmarks']['refined_2d'] = self.refine_2d_landmarks(
                            tddfa_result, delta_poses[idx], result['warped_landmarks']['projected_3d'],
                            result['warped_landmarks']['2d_style'][:17, :2], result['warped_landmarks']['3d_style'])
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

    @staticmethod
    def refine_2d_landmarks(tddfa_result: Dict, delta_pose: Sequence[float], landmarks: np.ndarray,
                            jaw_points_2d: np.ndarray, landmarks_3d: np.ndarray) -> np.ndarray:
        refined_landmarks = landmarks[:, :2].copy()
        pitch, yaw = tddfa_result['face_pose']['pitch'], tddfa_result['face_pose']['yaw']
        delta_pitch, delta_yaw = delta_pose[0], delta_pose[1]
        if delta_yaw != 0.0 or delta_pitch != 0.0:
            weight = np.clip((np.cos((yaw + delta_yaw) * 2.0) + 1.0) / (np.cos(yaw * 2.0) + 1.0),
                             0.0, 1.0)
            weight_p = np.clip((np.cos((pitch + delta_pitch) * 2.0) + 1.0) / (np.cos(pitch * 2.0) + 1.0),
                               0.0, 1.0)
            num_jaw_points = jaw_points_2d.shape[0]
            blending_indices = []
            bad_contour_indices = []
            contour_weights = np.full((num_jaw_points,), weight_p)
            if delta_yaw < 0.0 or delta_pitch != 0.0 and yaw < 0.0:
                if delta_yaw < 0.0:
                    blending_indices = list(range(17, 22)) + list(range(27, 34)) + [36, 39]
                contour_weights *= weight / (1.0 + np.exp(num_jaw_points // 2 - np.arange(num_jaw_points)))
                bad_contour_indices = [3, num_jaw_points // 2]
            elif delta_yaw > 0.0 or delta_pitch != 0.0 and yaw > 0.0:
                if delta_yaw > 0.0:
                    blending_indices = list(range(22, 31)) + list(range(33, 36)) + [42, 45]
                contour_weights *= weight / (1.0 + np.exp(np.arange(num_jaw_points) - num_jaw_points // 2))
                bad_contour_indices = [num_jaw_points // 2, num_jaw_points - 4]
            fixed_contour = jaw_points_2d.copy()
            if len(bad_contour_indices) > 0:
                good_contour_indices = [x for x in range(num_jaw_points) if x not in bad_contour_indices]
                interp_x = interp1d(good_contour_indices, jaw_points_2d[good_contour_indices, 0], kind='cubic')
                interp_y = interp1d(good_contour_indices, jaw_points_2d[good_contour_indices, 1], kind='cubic')
                fixed_contour[bad_contour_indices] = np.vstack((interp_x(bad_contour_indices),
                                                                interp_y(bad_contour_indices))).T
            if len(blending_indices) > 0:
                refined_landmarks[blending_indices] = (refined_landmarks[blending_indices] * weight +
                                                       landmarks_3d[blending_indices, :2] * (1.0 - weight))
            contour_weights = np.repeat(contour_weights, 2).reshape((-1, 2))
            refined_landmarks[:num_jaw_points] = (refined_landmarks[:num_jaw_points] * contour_weights +
                                                  fixed_contour * (1.0 - contour_weights))
            indices_for_smoothing = []
            if delta_yaw < 0.0 or delta_pitch != 0.0 and yaw < 0.0:
                indices_for_smoothing = range(num_jaw_points // 2 + 2, num_jaw_points - 1)
            elif delta_yaw > 0.0 or delta_pitch != 0.0 and yaw > 0.0:
                indices_for_smoothing = range(1, num_jaw_points // 2 - 1)
            if len(indices_for_smoothing) > 0:
                deltas = (refined_landmarks[indices_for_smoothing[0]: indices_for_smoothing[-1] + 2] -
                          refined_landmarks[indices_for_smoothing[0] - 1: indices_for_smoothing[-1] + 1])
                delta_norms = np.linalg.norm(deltas, axis=1)
                pseudo_curvature = (deltas[1:] * deltas[:-1]).sum(axis=1) / delta_norms[1:] / delta_norms[:-1]
                bad_index = np.argmin(pseudo_curvature) + indices_for_smoothing[0]
                good_indices = [x for x in range(num_jaw_points) if x != bad_index]
                interp_x = interp1d(good_indices, refined_landmarks[good_indices, 0], kind='cubic')
                interp_y = interp1d(good_indices, refined_landmarks[good_indices, 1], kind='cubic')
                updated_pt = np.hstack((interp_x(bad_index), interp_y(bad_index)))
                new_delta1 = updated_pt - refined_landmarks[bad_index - 1]
                new_delta2 = refined_landmarks[bad_index + 1] - updated_pt
                new_pc = np.dot(new_delta1, new_delta2) / np.linalg.norm(new_delta1) / np.linalg.norm(new_delta2)
                if new_pc > pseudo_curvature.min():
                    refined_landmarks[bad_index] = updated_pt

        return refined_landmarks
