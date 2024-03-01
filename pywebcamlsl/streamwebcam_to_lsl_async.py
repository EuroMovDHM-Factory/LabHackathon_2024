import cv2
import pylsl

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarkerResult

VisionRunningMode = mp.tasks.vision.RunningMode

num_poses = 4
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
video_source = 1

# Create a Lab Streaming Layer outlet
info = pylsl.StreamInfo("Pose", "Pose", 92, 0, "float32", "pose123")
outlet = pylsl.StreamOutlet(info)


def frame_callback(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    print(f"Pose landmarks: {result}")
    # Extract the pose landmarks as a list of floats
    pose_landmarks_list = result.pose_world_landmarks[0][10:]
    pose_data = []
    for landmark in pose_landmarks_list:
        pose_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

    # Stream the pose data over Lab Streaming Layer
    outlet.push_sample(pose_data)


base_options = python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_segmentation_masks=False,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    result_callback=frame_callback,
)
detector = vision.PoseLandmarker.create_from_options(options)


# Open the webcam video stream
cap = cv2.VideoCapture(video_source)

# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if not success:
        break

    # Convert the image to RGB format
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Process the image with mediapipe pose asynchronously
    detector.detect_async(image, frame_timestamp_ms)

# Release the webcam and close the window
cap.release()
