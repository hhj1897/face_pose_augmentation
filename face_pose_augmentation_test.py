import os
import cv2
import time
import torch
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor
from ibug.face_pose_augmentation import TDDFAPredictor


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', '-n', help='No display if processing a video file',
                        action='store_true', default=False)

    parser.add_argument('--detection-threshold', '-dt', type=float, default=0.8,
                        help='Confidence threshold for face detection (default=0.8)')
    parser.add_argument('--detection-method', '-dm', default='retinaface',
                        help='Face detection method, can be either RatinaFace or S3FD (default=RatinaFace)')
    parser.add_argument('--detection-weights', '-dw', default=None,
                        help='Weights to be loaded for face detection, ' +
                             'can be either resnet50 or mobilenet0.25 when using RetinaFace')
    parser.add_argument('--detection-device', '-dd', default='cuda:0',
                        help='Device to be used for face detection (default=cuda:0)')

    parser.add_argument('--alignment-threshold', '-at', type=float, default=0.2,
                        help='Score threshold used when visualising detected landmarks (default=0.2)'),
    parser.add_argument('--alignment-method', '-am', default='fan',
                        help='Face alignment method, must be set to FAN')
    parser.add_argument('--alignment-weights', '-aw', default=None,
                        help='Weights to be loaded for face alignment, can be either 2DFAN2 or 2DFAN4')
    parser.add_argument('--alignment-device', '-ad', default='cuda:0',
                        help='Device to be used for face alignment (default=cuda:0)')

    parser.add_argument('--tddfa-weights', '-tw', default=None,
                        help='Weights to be loaded by 3DDFA, must be set to mobilenet1')
    parser.add_argument('--tddfa-device', '-td', default='cuda:0',
                        help='Device to be used by 3DDFA.')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        args.detection_method = args.detection_method.lower()
        if args.detection_method == 'retinaface':
            face_detector = RetinaFacePredictor(threshold=args.detection_threshold, device=args.detection_device,
                                                model=(RetinaFacePredictor.get_model(args.detection_weights)
                                                       if args.detection_weights else None))
            print('Face detector created using RetinaFace.')
        elif args.detection_method == 's3fd':
            face_detector = S3FDPredictor(threshold=args.detection_threshold, device=args.detection_device,
                                          model=(S3FDPredictor.get_model(args.detection_weights)
                                                 if args.detection_weights else None))
            print('Face detector created using S3FD.')
        else:
            raise ValueError('detector-method must be set to either RetinaFace or S3FD')

        # Create the landmark detector
        args.alignment_method = args.alignment_method.lower()
        if args.alignment_method == 'fan':
            landmark_detector = FANPredictor(device=args.alignment_device,
                                             model=(FANPredictor.get_model(args.alignment_weights)
                                                    if args.alignment_weights else None))
            print('Landmark detector created using FAN.')
        else:
            raise ValueError('alignment-method must be set to FAN')

        # Instantiate 3DDFA
        tddfa = TDDFAPredictor(device=args.tddfa_device,
                               model=(TDDFAPredictor.get_model(args.tddfa_weights)
                                      if args.tddfa_weights else None))
        print('3DDFA initialised.')

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
            out_vid = cv2.VideoWriter(args.output, apiPreference=cv2.CAP_FFMPEG, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

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
                # Detect faces
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Face alignment
                start_time = current_time
                landmarks, scores = landmark_detector(frame, faces, rgb=False)
                current_time = time.time()
                elapsed_time2 = current_time - start_time

                ss = time.time()
                lala = TDDFAPredictor.decode(tddfa(frame, landmarks, rgb=False, two_steps=True))
                print(time.time() - ss)

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} + ' +
                      f'{elapsed_time2 * 1000.0:.04f} ms: {len(faces)} faces analysed..')

                # Rendering
                for face, yy in zip(faces, lala):
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
                    lm = tddfa.project_vertex(yy, False)
                    plot_landmarks(frame, lm[:, :2])
                    if len(face) > 5:
                        plot_landmarks(frame, face[5:].reshape((-1, 2)), pts_radius=3)

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
