class PoseCorrection:
    def __init__(self, metrics_dictionary: dict):
        self._metrics_dictionary = metrics_dictionary

    def correct_pose(self, landmarks: dict) -> dict:
        """
        Correct the pose based on the given landmarks.
        :param landmarks: Dictionary of landmarks.
        :return: Dictionary of corrected landmarks.
        """
        corrected_landmarks = dict()

        for frame_index in range(len(landmarks)):
            corrected_landmarks[frame_index] = self._metrics_dictionary(landmarks[frame_index])

        return corrected_landmarks
