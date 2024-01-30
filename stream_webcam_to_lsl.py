# Import the required modules
import cv2
import pylsl

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            list(solutions.pose.POSE_CONNECTIONS)[10:],
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


base_options = python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)


# Create a Lab Streaming Layer outlet
info = pylsl.StreamInfo("Pose", "Pose", 92, 0, "float32", "pose123")
outlet = pylsl.StreamOutlet(info)

# Open the webcam video stream
cap = cv2.VideoCapture(0)

# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Convert the image to RGB format
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Process the image with mediapipe pose
    results = detector.detect(image)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        # Show the image in a window)

        annotated_image = draw_landmarks_on_image(image.numpy_view(), results)
        cv2.imshow("Webcam Poses", annotated_image)

        # Extract the pose landmarks as a list of floats
        pose_landmarks_list = results.pose_world_landmarks[0][10:]
        pose_data = []
        for landmark in pose_landmarks_list:
            pose_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Stream the pose data over Lab Streaming Layer
        outlet.push_sample(pose_data)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
