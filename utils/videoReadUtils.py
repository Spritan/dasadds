import cv2
import mediapipe as mp
from typing import List, Tuple, Optional

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints_from_video(video_path:str) -> List[Optional[List[Tuple[float, float, float]]]]:
    """
    Extracts keypoints from a video using MediaPipe Pose.

    Parameters:
        video_path (str): Path to the input video.

    Returns:
        List[Optional[List[Tuple[float, float, float]]]]: List of keypoints for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect poses in the frame
        results = pose.process(frame_rgb)

        # Get keypoints
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        else:
            keypoints = None
        
        keypoints_list.append(keypoints)

    # Release VideoCapture object
    cap.release()

    return keypoints_list