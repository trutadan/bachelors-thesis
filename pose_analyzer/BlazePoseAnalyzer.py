import numpy as np

from constants import BLAZE_POSE_LANDMARKS
from pose_analyzer.PoseAnalyzer import PoseAnalyzer


class BlazePoseAnalyzer(PoseAnalyzer):
    def __init__(self):
        super().__init__(BLAZE_POSE_LANDMARKS)

    def compute_right_foot_knee_alignment(self, landmarks: dict) -> bool:
        """
        Compute the alignment of the right foot and knee.
        :param landmarks: Dictionary of landmarks.
        :return: True if the foot and knee are aligned, False otherwise.
        """
        right_hip = landmarks[self._key_points_dictionary['right hip']]
        right_knee = landmarks[self._key_points_dictionary['right knee']]
        right_ankle = landmarks[self._key_points_dictionary['right ankle']]
        right_foot_index = landmarks[self._key_points_dictionary['right foot index']]

        # create vectors
        knee_to_hip = np.array([right_hip[0] - right_knee[0], right_hip[1] - right_knee[1], right_hip[2] - right_knee[2]])
        ankle_to_foot_index = np.array([right_foot_index[0] - right_ankle[0], right_foot_index[1] - right_ankle[1], right_foot_index[2] - right_ankle[2]])

        # normalize vectors
        knee_to_hip_unit = knee_to_hip / np.linalg.norm(knee_to_hip)
        ankle_to_foot_index_unit = ankle_to_foot_index / np.linalg.norm(ankle_to_foot_index)

        # compute the dot product
        return np.dot(knee_to_hip_unit, ankle_to_foot_index_unit)

    def compute_left_foot_knee_alignment(self, landmarks: dict) -> bool:
        """
        Compute the alignment of the left foot and knee.
        :param landmarks: Dictionary of landmarks.
        :return: True if the foot and knee are aligned, False otherwise.
        """
        left_hip = landmarks[self._key_points_dictionary['left hip']]
        left_knee = landmarks[self._key_points_dictionary['left knee']]
        left_ankle = landmarks[self._key_points_dictionary['left ankle']]
        left_foot_index = landmarks[self._key_points_dictionary['left foot index']]

        # create vectors
        knee_to_hip = np.array([left_hip[0] - left_knee[0], left_hip[1] - left_knee[1], left_hip[2] - left_knee[2]])
        ankle_to_foot_index = np.array([left_foot_index[0] - left_ankle[0], left_foot_index[1] - left_ankle[1], left_foot_index[2] - left_ankle[2]])

        # normalize vectors
        knee_to_hip_unit = knee_to_hip / np.linalg.norm(knee_to_hip)
        ankle_to_foot_index_unit = ankle_to_foot_index / np.linalg.norm(ankle_to_foot_index)

        # compute the dot product
        return np.dot(knee_to_hip_unit, ankle_to_foot_index_unit)
