import pickle

from abc import ABC, abstractmethod


class LandmarksExtractor(ABC):
    def __init__(self):
        self._landmarks_dictionary = {}
        self._total_frames = 0

    def get_landmarks_dictionary(self) -> dict:
        return self._landmarks_dictionary

    def get_total_frames(self) -> int:
        return self._total_frames

    @abstractmethod
    def extract_landmarks(self, video_path: str) -> None:
        pass

    def save_landmarks(self, filepath: str) -> None:
        with open(filepath, 'wb') as file:
            pickle.dump(self._landmarks_dictionary, file)

    def load_landmarks(self, filepath: str) -> None:
        with open(filepath, 'rb') as file:
            self._landmarks_dictionary = pickle.load(file)
