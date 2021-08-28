
# MediaPipe: 0.8.7.1

import argparse

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# 参数设置
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


# 获取pose坐标
def mediapipe_pose_2D():
    # 获取参数
    args = get_args()

    # 相机设置
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    count = 0  # nose landmark 36

    with mp_pose.Pose(
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Print nose landmark.
            image_height, image_width, _ = image.shape
            if count < 36:
                count = count + 1
                # print("pose landmark:", results.pose_landmarks)
                print(
                    f'Nose coordinates: ('
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
                )

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()


def mediapipe_pose_3D():
    # 获取参数
    args = get_args()

    # 相机设置
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    with mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Print the real-world 3D coordinates of nose in meters with the origin at
            # the center between hips.
            print('Nose world landmark:'),
            print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])

            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()


if __name__ == '__main__':
    print("input 2 for 2D, 3 for 3D: ")
    flag = int(input())
    if flag == 2:
        mediapipe_pose_2D()
    elif flag == 3:
        mediapipe_pose_3D()
    else:
        print("input error.")