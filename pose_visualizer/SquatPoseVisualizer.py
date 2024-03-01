import mediapipe as mp

from pose_visualizer.PoseVisualizer import PoseVisualizer


class SquatPoseVisualizer(PoseVisualizer):
    def __init__(self, video_path, landmarks_dictionary):
        super().__init__(video_path, landmarks_dictionary)

    def draw_skeleton(self, image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    def draw_angles(self, image, landmarks):
        self.draw_angle_between_points(image, landmarks['left hip'], landmarks['left knee'], landmarks['left ankle'])
        self.draw_angle_between_points(image, landmarks['right hip'], landmarks['right knee'], landmarks['right ankle'])
        self.draw_angle_between_points(image, landmarks['left shoulder'], landmarks['left hip'], landmarks['left knee'])
        self.draw_angle_between_points(image, landmarks['right shoulder'], landmarks['right hip'], landmarks['right knee'])
