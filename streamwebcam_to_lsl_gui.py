import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QCheckBox,
    QSlider,
)
from PyQt5.QtCore import Qt


from mediapipe.tasks.python.vision import PoseLandmarkerResult

from pywebcamlsl.mediapipe import create_landmark_detector
from pywebcamlsl.pose_stream import PoseDataStreamer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap


class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.num_poses_label = QLabel("Number of poses")
        self.num_poses_input = QLineEdit("4")
        self.layout.addWidget(self.num_poses_label)
        self.layout.addWidget(self.num_poses_input)

        self.min_pose_detection_confidence_label = QLabel(
            "Minimum pose detection confidence"
        )
        self.min_pose_detection_confidence_input = QSlider(Qt.Horizontal)
        self.min_pose_detection_confidence_input.setMinimum(0)
        self.min_pose_detection_confidence_input.setMaximum(100)
        self.min_pose_detection_confidence_input.setValue(50)
        self.layout.addWidget(self.min_pose_detection_confidence_label)
        self.layout.addWidget(self.min_pose_detection_confidence_input)

        self.min_pose_presence_confidence_label = QLabel(
            "Minimum pose presence confidence"
        )
        self.min_pose_presence_confidence_input = QSlider(Qt.Horizontal)
        self.min_pose_presence_confidence_input.setMinimum(0)
        self.min_pose_presence_confidence_input.setMaximum(100)
        self.min_pose_presence_confidence_input.setValue(50)
        self.layout.addWidget(self.min_pose_presence_confidence_label)
        self.layout.addWidget(self.min_pose_presence_confidence_input)

        self.min_tracking_confidence_label = QLabel("Minimum tracking confidence")
        self.min_tracking_confidence_input = QSlider(Qt.Horizontal)
        self.min_tracking_confidence_input.setMinimum(0)
        self.min_tracking_confidence_input.setMaximum(100)
        self.min_tracking_confidence_input.setValue(50)
        self.layout.addWidget(self.min_tracking_confidence_label)
        self.layout.addWidget(self.min_tracking_confidence_input)

        self.video_source_label = QLabel("Video source")
        self.video_source_input = QLineEdit("0")
        self.layout.addWidget(self.video_source_label)
        self.layout.addWidget(self.video_source_input)

        self.model_path_label = QLabel("Model path")
        self.model_path_input = QLineEdit("pose_landmarker_heavy.task")
        self.layout.addWidget(self.model_path_label)
        self.layout.addWidget(self.model_path_input)

        self.stream_name_label = QLabel("Stream name")
        self.stream_name_input = QLineEdit("PoseData")
        self.layout.addWidget(self.stream_name_label)
        self.layout.addWidget(self.stream_name_input)

        self.start_button = QPushButton("Start Streaming")
        self.start_button.clicked.connect(self.start_streaming)
        self.layout.addWidget(self.start_button)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

    def start_streaming(self):
        num_poses = int(self.num_poses_input.text())
        min_pose_detection_confidence = (
            self.min_pose_detection_confidence_input.value() / 100
        )
        min_pose_presence_confidence = (
            self.min_pose_presence_confidence_input.value() / 100
        )
        min_tracking_confidence = self.min_tracking_confidence_input.value() / 100
        video_source = int(self.video_source_input.text())
        model_path = self.model_path_input.text()
        stream_name = self.stream_name_input.text()

        self.cap = cv2.VideoCapture(video_source)

        self.landmark_detector = create_landmark_detector(
            model_path,
            num_poses,
            min_pose_detection_confidence,
            min_pose_presence_confidence,
            min_tracking_confidence,
        )

        self.streamer = PoseDataStreamer(num_poses=num_poses, stream_name=stream_name)

        # Here you can call the function that starts the streaming with the parameters from the GUI
        # start_streaming(num_poses, min_pose_detection_confidence, min_pose_presence_confidence, min_tracking_confidence, video_source, asynchronous, model_path, stream_name)

    def update_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()

        if ret:
            # Convert the frame to RGB format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the RGB image to a QImage
            q_image = QImage(
                rgb_image.data,
                rgb_image.shape[1],
                rgb_image.shape[0],
                QImage.Format_RGB888,
            )

            # Convert the QImage to a QPixmap and display it in the QLabel
            self.image_label.setPixmap(QPixmap.fromImage(q_image))


if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()
