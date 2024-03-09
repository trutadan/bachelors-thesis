import numpy as np

from math import acos, degrees


class PoseAnalyzer:
    def __init__(self, key_points_dictionary: dict):
        self._key_points_dictionary = key_points_dictionary

    @staticmethod
    def compute_2d_angle(A: tuple, B: tuple, C: tuple) -> float:
        """
        Calculates the angle ABC (in degrees) in 2D between points A, B, and C.
        """
        BA = np.array([A[0] - B[0], A[1] - B[1]])
        BC = np.array([C[0] - B[0], C[1] - B[1]])

        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))

        # clip to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = degrees(acos(cosine_angle))

        return angle

    def compute_right_hip_knee_ankle_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the hip, knee, and ankle landmarks.
        """
        right_hip = landmarks[self._key_points_dictionary['right hip']]
        right_knee = landmarks[self._key_points_dictionary['right knee']]
        right_ankle = landmarks[self._key_points_dictionary['right ankle']]

        return self.compute_2d_angle(right_hip, right_knee, right_ankle)

    def compute_left_hip_knee_ankle_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the hip, knee, and ankle landmarks.
        """
        left_hip = landmarks[self._key_points_dictionary['left hip']]
        left_knee = landmarks[self._key_points_dictionary['left knee']]
        left_ankle = landmarks[self._key_points_dictionary['left ankle']]

        return self.compute_2d_angle(left_hip, left_knee, left_ankle)

    def compute_right_shoulder_hip_knee_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the shoulder, hip, and knee landmarks.
        """
        right_shoulder = landmarks[self._key_points_dictionary['right shoulder']]
        right_hip = landmarks[self._key_points_dictionary['right hip']]
        right_knee = landmarks[self._key_points_dictionary['right knee']]

        return self.compute_2d_angle(right_shoulder, right_hip, right_knee)

    def compute_left_shoulder_hip_knee_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the shoulder, hip, and knee landmarks.
        """
        left_shoulder = landmarks[self._key_points_dictionary['left shoulder']]
        left_hip = landmarks[self._key_points_dictionary['left hip']]
        left_knee = landmarks[self._key_points_dictionary['left knee']]

        return self.compute_2d_angle(left_shoulder, left_hip, left_knee)

    def compute_right_shoulder_elbow_wrist_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the shoulder, elbow, and wrist landmarks.
        """
        right_shoulder = landmarks[self._key_points_dictionary['right shoulder']]
        right_elbow = landmarks[self._key_points_dictionary['right elbow']]
        right_wrist = landmarks[self._key_points_dictionary['right wrist']]

        return self.compute_2d_angle(right_shoulder, right_elbow, right_wrist)

    def compute_left_shoulder_elbow_wrist_angle(self, landmarks: dict) -> float:
        """
        Computes the angle between the shoulder, elbow, and wrist landmarks.
        """
        left_shoulder = landmarks[self._key_points_dictionary['left shoulder']]
        left_elbow = landmarks[self._key_points_dictionary['left elbow']]
        left_wrist = landmarks[self._key_points_dictionary['left wrist']]

        return self.compute_2d_angle(left_shoulder, left_elbow, left_wrist)
