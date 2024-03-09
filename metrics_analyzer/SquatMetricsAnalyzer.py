from metrics_analyzer.MetricsAnalyzer import MetricsAnalyzer
from pose_analyzer.PoseAnalyzer import PoseAnalyzer
from pose_analyzer.BlazePoseAnalyzer import BlazePoseAnalyzer


class SquatMetricsAnalyzer(MetricsAnalyzer):
    def __init__(self, landmarks_dictionary: dict, pose_analyzer: PoseAnalyzer = BlazePoseAnalyzer()):
        super().__init__(landmarks_dictionary, pose_analyzer)
        self._metrics = {angle: {'values': [], 'statistics': {}} for angle in [
            'right_hip_knee_ankle', 'left_hip_knee_ankle', 'right_shoulder_hip_knee', 'left_shoulder_hip_knee'
        ]}
