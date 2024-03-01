import cv2

from pose_analyzer.PoseAnalyzer import PoseAnalyzer


class PoseVisualizer:
    def __init__(self, video_path, landmarks_dictionary):
        self._video_path = video_path
        self._landmarks_dictionary = landmarks_dictionary

    def visualize_video(self, show_angles: bool = False):
        cap = cv2.VideoCapture(self._video_path)
        frame_index = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success or frame_index not in self._landmarks_dictionary:
                break

            landmarks = self._landmarks_dictionary[frame_index]
            self.draw_skeleton(image, landmarks)

            if show_angles:
                self.draw_angles(image, landmarks)

            cv2.imshow('Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    def visualize_frame(self, frame_number: int, show_angles: bool = False):
        cap = cv2.VideoCapture(self._video_path)

        # set the frame to the specific frame number
        cap.set(1, frame_number)

        success, image = cap.read()
        if success and frame_number in self._landmarks_dictionary:
            landmarks = self._landmarks_dictionary[frame_number]
            self.draw_skeleton(image, landmarks)

            if show_angles:
                self.draw_angles(image, landmarks)

            cv2.imshow('Pose Frame', image)

            # wait indefinitely until a key is pressed
            cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def draw_angle_between_points(image, point1, point2, point3):
        """
        Draws the angle calculated between any three points on the image.
        :param image: The image on which to draw.
        :param point1: The first point (A) in the angle calculation.
        :param point2: The vertex of the angle (B).
        :param point3: The third point (C) in the angle calculation.
        """
        # calculate the angle
        angle = PoseAnalyzer.compute_2d_angle(point1, point2, point3)

        # convert the vertex point (point2) position to pixel coordinates for text placement
        point2_px = int(point2[0] * image.shape[1]), int(point2[1] * image.shape[0])

        # draw the angle text near the vertex point
        cv2.putText(image, f"{int(angle)} deg", point2_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_skeleton(self, image, landmarks):
        # Draw the skeleton using landmarks
        raise NotImplementedError("draw_skeleton method not implemented")

    def draw_angles(self, image, landmarks):
        # Draw the computed angles on the skeleton
        raise NotImplementedError("draw_angles method not implemented")
