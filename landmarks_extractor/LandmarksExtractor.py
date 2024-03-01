import pickle

from abc import ABC, abstractmethod


class LandmarksExtractor(ABC):
    def __init__(self):
        self._landmarks_dictionary = {}

    def get_landmarks_dictionary(self):
        return self._landmarks_dictionary

    @abstractmethod
    def extract_landmarks(self, video_path: str):
        pass

    def save_landmarks(self, filepath: str):
        with open(filepath, 'wb') as file:
            pickle.dump(self._landmarks_dictionary, file)

    def load_landmarks(self, filepath: str):
        with open(filepath, 'rb') as file:
            self._landmarks_dictionary = pickle.load(file)
