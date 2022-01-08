import os
import cv2
import time
import torch
import numpy as np
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_pose_augmentation import TDDFAPredictor, FacePoseAugmentor


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', '-n', help='No display if processing a video file',
                        action='store_true', default=False)
    parser.add_argument('--weights', '-w', default=None,
                        help='Weights to be loaded by 3DDFA, must be set to mobilenet1')
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='Device to be used by all models (default=cuda:0')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        face_detector = RetinaFacePredictor(device=args.device, model=RetinaFacePredictor.get_model('mobilenet0.25'))
        print('Face detector created.')

        # Create the landmark detector
        landmark_detector = FANPredictor(device=args.device, model=FANPredictor.get_model('2dfan2'))
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

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces and landmarks
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                landmarks, scores = landmark_detector(frame, faces, rgb=False)
                current_time = time.time()
                elapsed_time1 = current_time - start_time

                # Run TDDFA
                start_time = current_time
                tddfa_results = TDDFAPredictor.decode(tddfa(frame, landmarks, rgb=False, two_steps=True))
                current_time = time.time()
                elapsed_time2 = current_time - start_time

                # Select the largest valid face (in which all landmark scores >= 0.2) for pose augmentation
                selected_face_idx = -1
                if faces.shape[0] > 0:
                    valid_face_indices = np.where(scores.min(axis=-1) >= 0.2)[0]
                    if len(valid_face_indices) > 0:
                        valid_faces = faces[valid_face_indices]
                        valid_face_sizes = ((valid_faces[:, 2] - valid_faces[:, 0]) *
                                            (valid_faces[:, 3] - valid_faces[:, 1]))
                        selected_face_idx = valid_face_indices[np.argmax(valid_face_sizes)]

                # Render all other faces before pose augmentation
                for idx, (face, result) in enumerate(zip(faces, tddfa_results)):
                    if idx != selected_face_idx:
                        lms = tddfa.project_vertex(result, False)
                        plot_landmarks(frame, lms[:, :2], line_colour=(255, 0, 0), pts_colour=(255, 0, 255))
                        if len(face) > 5:
                            plot_landmarks(frame, face[5:].reshape((-1, 2)), pts_radius=3,
                                           line_colour=(255, 0, 0), pts_colour=(255, 0, 255))

                # Pose augmentation!
                start_time = time.time()
                if selected_face_idx >= 0:
                    print(tddfa_results[selected_face_idx]['face_pose']['pitch'],
                          tddfa_results[selected_face_idx]['face_pose']['yaw'])
                    augmentation_result = augmentor(frame, tddfa_results[selected_face_idx],
                                                    np.array([40.0, -45.0, 25.0]) / 180.0 * np.pi)
                                                    # landmarks[selected_face_idx])
                    frame = augmentation_result['warped_image']
                    plot_landmarks(frame, augmentation_result['warped_landmarks'][1, :2].T)
                current_time = time.time()
                elapsed_time3 = current_time - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time1 * 1000.0:.04f} + ' +
                      f'{elapsed_time2 * 1000.0:.04f} + ' + f'{elapsed_time3 * 1000.0:.04f} ms: ' +
                      f'{len(faces)} faces analysed..')

                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
