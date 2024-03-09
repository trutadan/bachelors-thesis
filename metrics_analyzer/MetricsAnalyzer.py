import pickle
import numpy as np

from pose_analyzer.PoseAnalyzer import PoseAnalyzer


class MetricsAnalyzer:
    def __init__(self, landmarks_dictionary: dict, pose_analyzer: PoseAnalyzer):
        self._landmarks_dictionary = landmarks_dictionary
        self._pose_analyzer = pose_analyzer
        self._metrics = {}

    def get_metrics(self) -> dict:
        return self._metrics

    def compute_metrics(self) -> None:
        """
        Computes metrics for all frames and stores both individual values and statistics.
        """
        for frame_index, landmarks in self._landmarks_dictionary.items():
            # Compute and store metrics for each frame
            for angle_name in self._metrics:
                angle_value = getattr(self._pose_analyzer, f'compute_{angle_name}_angle')(landmarks)
                self._metrics[angle_name]['values'].append((frame_index, angle_value))

        # Compute statistics for each angle
        self.__compute_statistics()

    def __compute_statistics(self) -> None:
        """
        Computes mean, max, and min for each angle based on the stored values.
        """
        for angle_name, data in self._metrics.items():
            values = [value for _, value in data['values']]
            data['statistics'] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
            }

    def save_metrics(self, filepath: str) -> None:
        """
        Saves the computed metrics to a pickle file.

        :param filepath: Path to the file where metrics should be saved.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self._metrics, file)

    def load_metrics(self, filepath: str) -> None:
        """
        Loads the metrics from a pickle file.

        :param filepath: Path to the file from which metrics should be loaded.
        """
        with open(filepath, 'rb') as file:
            self._metrics = pickle.load(file)
