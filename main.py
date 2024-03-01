from landmarks_extractor.BlazePoseLandmarksExtractor import BlazePoseLandmarksExtractor
from pose_analyzer.BlazePoseAnalyzer import BlazePoseAngleAnalyzer
from pose_visualizer.SquatPoseVisualizer import SquatPoseVisualizer


if __name__ == '__main__':
    landmarks_extractor = BlazePoseLandmarksExtractor()
    landmarks_extractor.extract_landmarks('videos/squat.mp4')
    landmarks_extractor.save_landmarks('data/landmarks/squat.pkl')

    blaze_pose_analyzer = BlazePoseAngleAnalyzer()

    squat_pose_visualizer = SquatPoseVisualizer('videos/squat.mp4', landmarks_extractor.get_landmarks_dictionary())
    squat_pose_visualizer.visualize_video(show_angles=True)
