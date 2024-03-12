import cv2


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from mediapipe.tasks.python.vision import PoseLandmarkerResult
import json

from pywebcamlsl.pose_stream import PoseDataStreamer
from pywebcamlsl.pose_view import draw_landmarks_on_image

VisionRunningMode = mp.tasks.vision.RunningMode

num_poses = 4
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
video_source = 0
asynchronous = True

landmark_start_index = 11
num_channels_per_landmark = 88

model_path = "pose_landmarker_heavy.task"

streamer = PoseDataStreamer(num_poses=4)

if asynchronous:
    running_mode = VisionRunningMode.LIVE_STREAM
else:
    running_mode = VisionRunningMode.IMAGE


def frame_callback(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    streamer.stream_pose_data(result, timestamp_ms)


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=running_mode,
    output_segmentation_masks=False,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

if asynchronous:
    options.result_callback = frame_callback
detector = vision.PoseLandmarker.create_from_options(options)


# Open the webcam video stream
cap = cv2.VideoCapture(video_source)

# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if success:
        # Convert the image to RGB format
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        if asynchronous:
            # Process the image with mediapipe pose asynchronously
            detector.detect_async(image, frame_timestamp_ms)
        else:
            result = detector.detect(image)
            if result.pose_landmarks:
                # Show the image in a window
                annotated_image = draw_landmarks_on_image(image.numpy_view(), result)
                cv2.imshow("Webcam Poses", annotated_image)
                streamer.stream_pose_data(result, frame_timestamp_ms)

    else:
        print("Failed to read frame")

# Release the webcam and close the window
cap.release()
