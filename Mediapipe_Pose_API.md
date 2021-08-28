# MediaPipe Pose API

[MediaPipe_Pose_Python_API](https://google.github.io/mediapipe/solutions/pose#python-solution-api)

[mediapipe pose Colab](https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy)



**code:**

```python
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

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
```

**配置选项：**

(不同平台不同语言的命名可能不同)

- [static_image_mode](https://google.github.io/mediapipe/solutions/pose#static_image_mode)

如果设置为 `false`，该解决方案会将输入图像视为视频流(video stream)。它将尝试在第一张图像中检测最突出的人，并在成功检测后进一步定位姿势地标。在随后的图像中，它只是简单地跟踪那些地标，而不会调用另一个检测，直到它失去跟踪，以减少计算和延迟。如果设置为 `true`，则人员检测会运行每个输入图像，非常适合处理一批静态的、可能不相关的图像。默认为`false`。

- [model_complexity](https://google.github.io/mediapipe/solutions/pose#model_complexity)

姿势地标模型的复杂度：`0`、`1 `或` 2`。地标准确度和推理延迟通常随着模型复杂度的增加而增加。默认为 `1`。

- [smooth_landmarks](https://google.github.io/mediapipe/solutions/pose#smooth_landmarks)

如果设置为 `true`，解决方案会过滤不同输入图像之间的地标以减少抖动(jitter)，但如果 `static_image_mode` 也设置为 `true`，则忽略。默认为`true`。

- [enable_segmentation](https://google.github.io/mediapipe/solutions/pose#enable_segmentation)

如果设置为 `true`，除了姿势地标之外，该解决方案还会生成分割掩码(segmentation mask)。默认为`false`。

- [smooth_segmentation](https://google.github.io/mediapipe/solutions/pose#smooth_segmentation)

如果设置为 `true`，该解决方案会过滤不同输入图像的分割掩码以减少抖动。如果 `enable_segmentation` 为 `false` 或 `static_image_mode` 为 `true`，则忽略。默认为`false`。

- [min_detection_confidence](https://google.github.io/mediapipe/solutions/pose#min_detection_confidence)、

来自人员检测模型的最小置信值 (`[0.0, 1.0]`)，用于将检测视为成功。默认为 `0.5`。

- [min_tracking_confidence](https://google.github.io/mediapipe/solutions/pose#min_tracking_confidence)

来自地标跟踪模型的最小置信值 (`[0.0, 1.0]`)，用于被视为成功跟踪的姿势地标，否则将在下一个输入图像上自动调用人物检测。将其设置为更高的值可以提高解决方案的稳健性，但代价是更高的延迟。如果 `static_image_mode` 为 `true`，则忽略，其中人员检测仅在每个图像上运行。默认为 `0.5`。



**输出：**

(不同平台不同语言的命名可能不同)

- `pose_landmarks`

姿势地标列表。每个地标包括以下内容：

- `x` 和 `y`：分别由图像宽度和高度归一化为 `[0.0, 1.0]` 的地标坐标。
- `z`：代表地标深度，以**臀部中点**的深度为原点，值越小，地标离相机越近。 `z`的大小使用与 `x` 大致相同的比例。
- `visibility`：`[0.0, 1.0]` 中的值指示地标在图像中可见（存在且未被遮挡）的可能性。



- `pose_world_landmarks`

世界坐标中的另一个姿势地标列表。每个地标包括以下内容：

`x`、`y `和` z`：以米为单位的真实世界 3D 坐标，原点位于臀部之间的中心。
`visibility`：与相应`pose_landmarks`中定义的相同。

https://google.github.io/mediapipe/images/mobile/pose_world_landmarks.mp4



- `segmentation_mask`

输出分段掩码，仅在 `enable_segmentation` 设置为 `true` 时预测。掩码与输入图像具有相同的宽度和高度，并包含 `[0.0, 1.0]` 中的值，其中 `1.0` 和 `0.0` 分别表示“人类”和“背景”像素的高确定性。有关使用详情，请参阅以下特定于平台的使用示例。

https://google.github.io/mediapipe/images/mobile/pose_segmentation.mp4

