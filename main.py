import numpy as np
from moviepy.editor import VideoFileClip


from landmarks_extractor.BlazePoseLandmarksExtractor import BlazePoseLandmarksExtractor
from metrics_analyzer.SquatMetricsAnalyzer import SquatMetricsAnalyzer
from pose_analyzer.BlazePoseAnalyzer import BlazePoseAnalyzer
from pose_visualizer.SquatPoseVisualizer import SquatPoseVisualizer


def smooth_data(data, window_length=3):
    """Applies a simple moving average to smooth the data."""
    return np.convolve(data, np.ones(window_length) / window_length, mode='valid')


def find_repetitions(angle_data, threshold=10):
    """Finds repetitions based on significant angle changes."""
    frames, angles = zip(*angle_data)
    smoothed_angles = smooth_data(np.array(angles))

    # Initialize variables
    repetitions = []
    direction = 0  # -1 for down, 1 for up
    start_frame = None

    for i in range(1, len(smoothed_angles) - 1):
        angle_change = smoothed_angles[i] - smoothed_angles[i - 1]

        # Detect change in direction
        if angle_change > threshold:
            current_direction = 1  # Moving up
        elif angle_change < -threshold:
            current_direction = -1  # Moving down
        else:
            continue

        # Check for repetition start (downward movement begins)
        if current_direction != direction and current_direction == -1:
            start_frame = frames[i]
            direction = current_direction

        # Check for repetition end (upward movement begins)
        elif current_direction != direction and current_direction == 1 and start_frame is not None:
            end_frame = frames[i]
            repetitions.append((start_frame, end_frame))
            start_frame = None  # Reset for next repetition
            direction = current_direction

    return repetitions


def split_video_moviepy(video_path, repetitions, output_folder):
    video = VideoFileClip(video_path)
    for i, (start, end) in enumerate(repetitions, start=1):
        # Convert start and end times from frames to seconds if needed
        start_sec = start / video.fps
        end_sec = end / video.fps
        subclip = video.subclip(start_sec, end_sec)
        subclip.write_videofile(f"{output_folder}/squat_{i}.mp4", codec="libx264")


if __name__ == '__main__':
    landmarks_extractor = BlazePoseLandmarksExtractor()
    # landmarks_extractor.extract_landmarks('videos/squat.mp4')
    # landmarks_extractor.save_landmarks('data/landmarks/squat.pkl')

    landmarks_extractor.load_landmarks('data/landmarks/squat.pkl')

    blaze_pose_analyzer = BlazePoseAnalyzer()

    squat_pose_visualizer = SquatPoseVisualizer('videos/squat.mp4', landmarks_extractor.get_landmarks_dictionary())
    squat_pose_visualizer.visualize_video(show_angles=True)
    # squat_pose_visualizer.visualize_frame(35, True)

    metrics_analyzer = SquatMetricsAnalyzer(landmarks_extractor.get_landmarks_dictionary(), blaze_pose_analyzer)
    metrics_analyzer.compute_metrics()
    metrics_analyzer.save_metrics("data/metrics/squat.pkl")
    print(metrics_analyzer.get_metrics())

    dicti = metrics_analyzer.get_metrics()['right_hip_knee_ankle']['values']
    print(dicti)
    print(find_repetitions(dicti, landmarks_extractor.get_total_frames()))
    # split_video_moviepy('videos/squat.mp4', find_repetitions(dicti, landmarks_extractor.get_total_frames()), 'videos')
