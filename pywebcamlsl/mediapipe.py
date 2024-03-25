import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_landmark_detector(
    model_path="pose_landmarker_heavy.task",
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        output_segmentation_masks=False,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return vision.PoseLandmarker.create_from_options(options)
