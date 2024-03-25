from mediapipe.tasks.python.vision import PoseLandmarkerResult
import json

import pylsl


class PoseDataStreamer:
    def __init__(
        self,
        num_poses,
        landmark_start_index=11,
        num_channels_per_landmark=88,
        model_path="pose_landmarker_heavy.task",
        stream_name="Pose",
    ):
        self.landmark_start_index = landmark_start_index
        self.num_channels_per_landmark = num_channels_per_landmark
        self.model_path = model_path
        self.num_poses = num_poses
        self.stream_name = stream_name

        self.body_parts = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        info = pylsl.StreamInfo(
            self.stream_name,
            "Pose",
            self.num_channels_per_landmark * self.num_poses,
            0,
            "float32",
            "PoseData",
        )

        # Create the channel names
        channel_names = []
        for i in range(self.num_poses):
            for part in self.body_parts:
                for coord in ["_x", "_y", "_z", "_visibility"]:
                    channel_names.append(f"{part}{coord}_{i+1}")

        # Set the channel names in the LSL stream info
        chns = info.desc().append_child("channels")
        for label in channel_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label).append_child_value(
                "type", "PoseData"
            ).append_child_value("unit", "m")

        meta = info.desc().append_child("metadata")
        meta.append_child_value("manufacturer", "Mediapipe")
        meta.append_child_value("model", f"PoseLandmarker ({self.model_path})")
        meta.append_child_value("num_poses", f"{self.num_poses}")

        self.outlet = pylsl.StreamOutlet(info)

    def stream_pose_data(self, result: PoseLandmarkerResult, timestamp: float):
        # Prettyprint result __dict__ with indents
        # print(result)

        pose_data = []
        if result.pose_world_landmarks is not None:
            # print(len(result.pose_world_landmarks))
            for skeleton in result.pose_world_landmarks:
                if isinstance(skeleton, list):
                    pose_landmarks_list = skeleton[self.landmark_start_index :]
                else:
                    pose_landmarks_list = skeleton.landmark[self.landmark_start_index :]
                for landmark in pose_landmarks_list:
                    pose_data.extend(
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    )

        num_detected_poses = len(result.pose_world_landmarks)
        if num_detected_poses < self.num_poses:
            pose_data.extend(
                [0.0]
                * self.num_channels_per_landmark
                * (self.num_poses - num_detected_poses)
            )
        # Stream the pose data over Lab Streaming Layer
        self.outlet.push_sample(pose_data, timestamp)
