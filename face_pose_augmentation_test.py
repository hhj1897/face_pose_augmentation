import os
import cv2
import time
import torch
import numpy as np
from typing import Tuple, Union
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_pose_augmentation import TDDFAPredictor, FacePoseAugmentor


def face_detection_loop(vid: cv2.VideoCapture, face_detector: RetinaFacePredictor,
                        landmark_detector: FANPredictor, window_title: str) \
        -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    print('Face and landmark detection started, you can use the following commands:\n' +
          '  |_C: Capture the selected face for pose augmentation.\n'
          '  |_Q: Quit the demo.')
    while True:
        _, frame = vid.read()

        # Face and landmark detection
        faces = face_detector(frame, rgb=False)
        landmarks, scores = landmark_detector(frame, faces, rgb=False)

        # Try to select a face
        face_sizes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if scs.min() >= 0.2 else -1
                      for bbox, scs in zip(faces, scores)]
        selected_face = np.argmin(face_sizes) if len(face_sizes) > 0 and min(face_sizes) > 0 else -1

        # Visualisation
        for idx, (lms, scs) in enumerate(zip(landmarks, scores)):
            if idx != selected_face:
                plot_landmarks(frame, lms, scs, line_colour=(192, 192, 192), pts_colour=(128, 128, 128))
        if selected_face < 0:
            frame_vis = frame
        else:
            frame_vis = frame.copy()
            plot_landmarks(frame_vis, landmarks[selected_face], scores[selected_face])

        # Show the frame and process commands
        cv2.imshow(window_title, frame_vis)
        key = cv2.waitKey(1) % 2 ** 16
        if key == ord('c') or key == ord('C'):
            if selected_face < 0:
                print('\'C\' pressed, but there is no face being selected.')
            else:
                print('\'C\' pressed, applying pose augmentation to the selected face.')
                return frame, landmarks[selected_face]
        elif key == ord('q') or key == ord('Q'):
            print('\'Q\' pressed, we are done here.')
            return None, None


def face_pose_augmentation_loop(tddfa: TDDFAPredictor, augmentor: FacePoseAugmentor,
                                frame: np.ndarray, landmarks: np.ndarray, landmark_style_index: int,
                                window_title: str) -> int:
    # Apply 3DDFA
    start_time = time.time()
    tddfa_result = TDDFAPredictor.decode(tddfa(frame, landmarks, rgb=False))[0]
    pitch, yaw, roll = np.array([tddfa_result['face_pose'][k] for k in ('pitch', 'yaw', 'roll')]) / np.pi * 180.0
    print(f'3D face model fitted in {(time.time() - start_time) * 1000.0:.3f} ms.')
    print(f'The estimated head pose (pitch, yaw, and roll, in degree) is ({pitch:.3f}, {yaw:.3f}, {roll:.3f})')

    # Determine delta poses
    delta_poses = []
    delta_pitchs = np.arange(-20, 21, 10)
    delta_yaws = np.arange(0, -90 - yaw, -10) if yaw < 0 else np.arange(0, 90 - yaw, 10)
    for dp in delta_pitchs:
        for dy in delta_yaws:
            delta_poses.append((dp, dy, 0))
    delta_poses = np.array(delta_poses) / 180.0 * np.pi

    # Pose augmentation
    start_time = time.time()
    augmentation_results = augmentor(frame, tddfa_result, delta_poses, landmarks)
    print(f'Pose augmentation finished in {(time.time() - start_time):.3f} second.')

    # Display the result
    dp_idx, dy_idx = len(delta_pitchs) // 2, 0
    landmark_styles = ['3d_style', '2d_style', 'projected_2d']
    if landmark_style_index > 0:
        print(f'Displaying result with \'{landmark_styles[landmark_style_index - 1]}\' landmarks, ' +
              'you can use the following commands:')
    else:
        print('Displaying result with no landmarks, you can use the following commands:')
    print('  |_A: Turn left (decrease yaw).\n' +
          '  |_D: Turn right (increase yaw).\n' +
          '  |_W: Tilt up (decrease pitch).\n' +
          '  |_S: Tilt down (increase pitch).\n' +
          '  |_0: Do not display landmarks.\n' +
          f'  |_1: Display \'{landmark_styles[0]}\' landmarks\n' +
          f'  |_2: Display \'{landmark_styles[1]}\' landmarks\n' +
          f'  |_3: Display \'{landmark_styles[2]}\' landmarks\n' +
          '  |_C: Goes back to face and landmark detection.\n' +
          '  |_Q: Quit the demo.')
    while True:
        result = augmentation_results[dp_idx * len(delta_yaws) + dy_idx]
        if landmark_style_index > 0:
            frame_vis = result['warped_image'].copy()
            plot_landmarks(frame_vis, result['warped_landmarks'][landmark_styles[landmark_style_index - 1]][:, :2])
        else:
            frame_vis = result['warped_image']
        cv2.imshow(window_title, frame_vis)
        key = cv2.waitKey(0) % 2 ** 16
        if key == ord('a') or key == ord('A'):
            dy_idx = min(dy_idx + 1, len(delta_yaws) - 1) if yaw < 0 else max(0, dy_idx - 1)
            print(f'\'A\' pressed: turing left by setting head pose to ({pitch:.3f}, {yaw:.3f}, {roll:.3f}) + ' +
                  f'({delta_pitchs[dp_idx]:.1f}, {delta_yaws[dy_idx]:.1f}, 0.0)')
        elif key == ord('d') or key == ord('D'):
            dy_idx = max(0, dy_idx - 1) if yaw < 0 else min(dy_idx + 1, len(delta_yaws) - 1)
            print(f'\'D\' pressed: turing right by setting head pose to ({pitch:.3f}, {yaw:.3f}, {roll:.3f}) + ' +
                  f'({delta_pitchs[dp_idx]:.1f}, {delta_yaws[dy_idx]:.1f}, 0.0)')
        elif key == ord('w') or key == ord('W'):
            dp_idx = max(0, dp_idx - 1)
            print(f'\'W\' pressed: tilting up by setting head pose to ({pitch:.3f}, {yaw:.3f}, {roll:.3f}) + ' +
                  f'({delta_pitchs[dp_idx]:.1f}, {delta_yaws[dy_idx]:.1f}, 0.0)')
        elif key == ord('s') or key == ord('S'):
            dp_idx = min(dp_idx + 1, len(delta_pitchs) - 1)
            print(f'\'S\' pressed: tilting down by setting head pose to ({pitch:.3f}, {yaw:.3f}, {roll:.3f}) + ' +
                  f'({delta_pitchs[dp_idx]:.1f}, {delta_yaws[dy_idx]:.1f}, 0.0)')
        elif ord('0') <= key <= ord('3'):
            landmark_style_index = key - ord('0')
            if landmark_style_index > 0:
                print(f'\'{chr(key)}\' pressed, setting to display ' +
                      f'\'{landmark_styles[landmark_style_index - 1]}\' landmarks.')
            else:
                print(f'\'{chr(key)}\' pressed, setting to not display landmarks.')
        elif key == ord('c') or key == ord('C'):
            print('\'C\' pressed, going back to face and landmark detection.')
            return landmark_style_index
        elif key == ord('q') or key == ord('Q'):
            print('\'Q\' pressed, we are done here.')
            return -1


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--weights', '-w', default=None,
                        help='Weights to be loaded by 3DDFA, must be set to mobilenet1')
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='Device to be used by all models (default=cuda:0')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    try:
        # Create the face detector
        face_detector = RetinaFacePredictor(device=args.device)
        print('Face detector created.')

        # Create the landmark detector
        landmark_detector = FANPredictor(device=args.device, model=FANPredictor.get_model('2dfan2_alt'))
        print('Landmark detector created.')

        # Instantiate 3DDFA
        tddfa = TDDFAPredictor(device=args.device, model=(TDDFAPredictor.get_model(args.weights)
                                                          if args.weights else None))
        print('3DDFA initialised.')

        # Create the face pose augmentor
        augmentor = FacePoseAugmentor()
        print('Face pose augmentor created.')

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # The main processing loop
        landmark_style_index = 1
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        while True:
            frame, landmarks = face_detection_loop(vid, face_detector, landmark_detector, window_title)
            cv2.destroyAllWindows()
            if frame is None or landmarks is None:
                break
            else:
                landmark_style_index = face_pose_augmentation_loop(tddfa, augmentor, frame, landmarks,
                                                                   landmark_style_index, window_title)
                if landmark_style_index < 0:
                    break
    finally:
        cv2.destroyAllWindows()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
