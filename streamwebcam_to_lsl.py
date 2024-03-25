import argparse
import cv2


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from mediapipe.tasks.python.vision import PoseLandmarkerResult
import numpy as np

from pywebcamlsl.pose_stream import PoseDataStreamer
from pywebcamlsl.pose_view import draw_landmarks_on_image


# Create the parser
parser = argparse.ArgumentParser(description="Stream webcam to LSL")

# Add the arguments
parser.add_argument("--num_poses", type=int, default=4, help="Number of poses")
parser.add_argument(
    "--min_pose_detection_confidence",
    type=float,
    default=0.5,
    help="Minimum pose detection confidence",
)
parser.add_argument(
    "--min_pose_presence_confidence",
    type=float,
    default=0.5,
    help="Minimum pose presence confidence",
)
parser.add_argument(
    "--min_tracking_confidence",
    type=float,
    default=0.5,
    help="Minimum tracking confidence",
)
parser.add_argument("--video_source", type=int, default=0, help="Video source")
parser.add_argument("--asynchronous", type=bool, default=True, help="Asynchronous mode")
parser.add_argument(
    "--model_path", type=str, default="pose_landmarker_heavy.task", help="Model path"
)

parser.add_argument("--stream-name", type=str, default="Pose", help="Stream name")

args = parser.parse_args()


VisionRunningMode = mp.tasks.vision.RunningMode
# Use the arguments
num_poses = args.num_poses
min_pose_detection_confidence = args.min_pose_detection_confidence
min_pose_presence_confidence = args.min_pose_presence_confidence
min_tracking_confidence = args.min_tracking_confidence
video_source = args.video_source
asynchronous = args.asynchronous
model_path = args.model_path
stream_name = args.stream_name

landmark_start_index = 11
num_channels_per_landmark = 88


streamer = PoseDataStreamer(num_poses=num_poses, stream_name=stream_name)

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

success, frame = cap.read()
frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
if success:
    # Convert the image to RGB format
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

cv2_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2RGBA)
cv2.imwrite(f"control_cam{video_source}_{stream_name}.png", cv2_image)

cv2.imshow("Camera Check, Press any key to continue", cv2_image)
# Wait for any key press
cv2.waitKey(0)

cv2.destroyAllWindows()

print("Starting Streaming...")

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
