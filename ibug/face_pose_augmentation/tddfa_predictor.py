import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
from types import SimpleNamespace
from typing import Union, Optional, List, Dict
from .tddfa import mobilenet_v1
from .tddfa.utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts
from .tddfa_utils import parse_param_pose, reconstruct_from_3dmm


__all__ = ['TDDFAPredictor']


class TDDFAPredictor(object):
    tri = loadmat(os.path.join(os.path.dirname(__file__), 'tddfa', 'visualize', 'tri.mat'))['tri']

    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None,
                 config: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = TDDFAPredictor.get_model()
        if config is None:
            config = TDDFAPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = getattr(mobilenet_v1, self.config.arch)(num_classes=62).to(self.device)
        pretrained_dict = torch.load(model.weights, map_location=self.device)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = {key.split('module.', 1)[-1] if key.startswith('module.') else key: value
                               for key, value in pretrained_dict['state_dict'].items()}
        else:
            pretrained_dict = {key.split('module.', 1)[-1] if key.startswith('module.') else key: value
                               for key, value in pretrained_dict.items()}
        self.net.load_state_dict(pretrained_dict)
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(1, 3, self.config.input_size,
                                                            self.config.input_size).to(self.device))

    @staticmethod
    def get_model(name: str = 'mobilenet1') -> SimpleNamespace:
        name = name.lower()
        if name == 'mobilenet1':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(mobilenet_v1.__file__),
                                                        'models', 'phase1_wpdc_vdc.pth.tar'),
                                   config=SimpleNamespace(arch='mobilenet_1', input_size=120))
        else:
            raise ValueError('name must be set to mobilenet')

    @staticmethod
    def create_config(use_jit: bool = True) -> SimpleNamespace:
        return SimpleNamespace(use_jit=use_jit)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, landmarks: np.ndarray, rgb: bool = True,
                 two_steps: bool = False) -> np.ndarray:
        if landmarks.size > 0:
            # Preparation
            if rgb:
                image = image[..., ::-1]
            if landmarks.ndim == 2:
                landmarks = landmarks[np.newaxis, ...]

            # Crop the face patches
            roi_boxes = []
            face_patches = []
            for lms in landmarks:
                roi_boxes.append(parse_roi_box_from_landmark(lms.T))
                face_patches.append(cv2.resize(crop_img(image, roi_boxes[-1]),
                                               (self.config.input_size, self.config.input_size)))
            face_patches = (torch.from_numpy(np.array(face_patches).transpose(
                (0, 3, 1, 2)).astype(np.float32)).to(self.device) - 127.5) / 128.0

            # Get 3DMM parameters
            params = self.net(face_patches).cpu().numpy()
            if two_steps:
                landmarks = []
                for param, roi_box in zip(params, roi_boxes):
                    landmarks.append(predict_68pts(param, roi_box).T)
                return self.__call__(image, np.array(landmarks), rgb=False, two_steps=False)
            else:
                return np.hstack((np.array(roi_boxes, dtype=np.float32), params))
        else:
            return np.empty(shape=(0, 66), dtype=np.float32)

    @staticmethod
    def decode(tdmm_params: np.ndarray) -> List[Dict]:
        if tdmm_params.size > 0:
            if tdmm_params.ndim > 1:
                return [TDDFAPredictor.decode(x) for x in tdmm_params]
            else:
                roi_box = tdmm_params[:4]
                params = tdmm_params[4:]
                vertex, pts68, fR, T = reconstruct_from_3dmm(params)
                camera_transform = {'fR': fR, 'T': T}
                yaw, pitch, roll, t3d, f = parse_param_pose(params)
                face_pose = {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't3d': t3d, 'f': f}
                return {'roi_box': roi_box, 'params': params, 'vertex': vertex, 'pts68': pts68,
                        'face_pose': face_pose, 'camera_transform': camera_transform}
        else:
            return []

    def project_vertex(self, tddfa_result: Dict, dense: bool = True) -> np.ndarray:
        vertex = (tddfa_result['camera_transform']['fR'] @
                  (tddfa_result['vertex'] if dense else tddfa_result['pts68']) +
                  tddfa_result['camera_transform']['T'])

        sx, sy, ex, ey = tddfa_result['roi_box']
        scale_x = (ex - sx) / self.config.input_size
        scale_y = (ey - sy) / self.config.input_size
        vertex[0, :] = vertex[0, :] * scale_x + sx
        vertex[1, :] = (self.config.input_size + 1 - vertex[1, :]) * scale_y + sy

        s = (scale_x + scale_y) / 2
        vertex[2, :] *= s

        return vertex.T
