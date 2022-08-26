import os
import cv2
import glob
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_pose_augmentation import TDDFAPredictor, FacePoseAugmentor


def detect_face(
    image: np.ndarray,
    face_detector: RetinaFacePredictor,
    landmark_detector: FANPredictor,
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    # Face and landmark detection
    faces = face_detector(image, rgb=False)
    landmarks, scores = landmark_detector(image, faces, rgb=False)

    # Try to select a face
    face_sizes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if scs.min() >= 0.2 else -1
                    for bbox, scs in zip(faces, scores)]
    selected_face = np.argmin(face_sizes) if len(face_sizes) > 0 and min(face_sizes) > 0 else -1

    return landmarks[selected_face] if selected_face >= 0 else None


def augment_face_pose(
    tddfa: TDDFAPredictor,
    augmentor: FacePoseAugmentor,
    image: np.ndarray,
    landmarks: np.ndarray,
    delta_pose: np.ndarray, # poses are in degrees
) -> int:
    # Apply 3DDFA
    tddfa_result = TDDFAPredictor.decode(tddfa(image, landmarks, rgb=False))[0]
    pitch, yaw, roll = np.array([tddfa_result['face_pose'][k] for k in ('pitch', 'yaw', 'roll')]) / np.pi * 180.0

    # Determine delta poses
    delta_pose = delta_pose / 180.0 * np.pi

    # Pose augmentation
    augmentation_results = augmentor(image, tddfa_result, delta_pose, landmarks)

    return augmentation_results[0]


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input-folder', '-i', default='samples/images',
                        help='Input folder of images')
    parser.add_argument('--output-folder', '-o', default='samples/outputs',
                        help='Output folder for the fitting results')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--weights', '-w', default=None,
                        help='Weights to be loaded by 3DDFA, must be set to mobilenet1')
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='Device to be used by all models (default=cuda:0')
    parser.add_argument('--plot-landmarks', help='Plot refined 2D benchmarks on the augmented image',
                        action='store_true', default=False)

    parser.add_argument('--pitch', '-p', default=0, type=float,
                        help='Delta of Pitch (in degree) for pose augmentation')
    parser.add_argument('--yaw', '-y', default=0, type=float,
                        help='Delta of Yaw (in degree) for pose augmentation')
    parser.add_argument('--roll', '-r', default=0, type=float,
                        help='Delta of Roll (in degree) for pose augmentation')

    parser.add_argument('--alignment-weights', '-aw', default='2dfan2_alt',
                        help='Weights to be loaded for face alignment, can be either 2DFAN2, 2DFAN4, ' +
                             'or 2DFAN2_ALT (default=2DFAN2_ALT)')
    parser.add_argument('--alignment-alternative-pth', '-ap', default=None,
                        help='Alternative pth file to be loaded for face alignment')
    parser.add_argument('--alignment-alternative-landmarks', '-al', default=None,
                        help='Alternative number of landmarks to detect')
                
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    # Create the face detector
    face_detector = RetinaFacePredictor(device=args.device)
    print('Face detector created.')

    # Create the landmark detector
    if args.alignment_weights is None:
        fa_model = FANPredictor.get_model()
    else:
        fa_model = FANPredictor.get_model(args.alignment_weights)

    if args.alignment_alternative_pth is not None:
        fa_model.weights = args.alignment_alternative_pth

    if args.alignment_alternative_landmarks is not None:
        fa_model.config.num_landmarks = int(args.alignment_alternative_landmarks)

    landmark_detector = FANPredictor(device=args.device, model=fa_model)
    print('Landmark detector created.')

    # Instantiate 3DDFA
    tddfa = TDDFAPredictor(device=args.device, model=(TDDFAPredictor.get_model(args.weights)
                                                        if args.weights else None))
    print('3DDFA initialised.')

    # Create the face pose augmentor
    augmentor = FacePoseAugmentor()
    print('Face pose augmentor created.')

    # create the output folder
    os.makedirs(args.output_folder, exist_ok=True)

    delta_pose = np.array([[args.pitch, args.yaw, args.roll]])
    image_files = sorted(glob.glob(os.path.join(args.input_folder, '*')))
    for image_fi in tqdm(image_files):
        # load the image
        try:
            image = cv2.imread(image_fi)
        except:
            print('Failed to load ', image_fi)
            continue

        # The main processing loop
        landmarks = detect_face(image, face_detector, landmark_detector)
        if landmarks is None:
            print('Failed to detect landmarks from ', image_fi)
            continue

        # augment the face image
        result = augment_face_pose(tddfa, augmentor, image, landmarks, delta_pose)

        # We will not save the correspondence_map in this script
        if "correspondence_map" in result:
            del result["correspondence_map"]
        # save the image separately
        filename = os.path.split(image_fi)[-1]
        # plot the landmarks
        warped_image = result["warped_image"].copy()
        if args.plot_landmarks:
            plot_landmarks(warped_image, result['warped_landmarks']['refined_2d'])
        cv2.imwrite(os.path.join(args.output_folder, filename), warped_image)
        del result["warped_image"]

        pickle.dump(result, open(os.path.join(args.output_folder, Path(filename).stem+'.pkl'),'wb'))

    print('All done.')


if __name__ == '__main__':
    main()
