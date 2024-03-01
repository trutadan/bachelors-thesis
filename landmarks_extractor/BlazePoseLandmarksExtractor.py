import cv2
import mediapipe as mp

from landmarks_extractor.LandmarksExtractor import LandmarksExtractor


class BlazePoseLandmarksExtractor(LandmarksExtractor):
    def __init__(self):
        super().__init__()
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, video_path: str):
        self._landmarks_dictionary = dict()

        cap = cv2.VideoCapture(video_path)
        frame_index = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
                self._landmarks_dictionary[frame_index] = landmarks

            frame_index += 1

        cap.release()
