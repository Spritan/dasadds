import numpy as np
import mediapipe as mp
from numpy.linalg import norm

import streamlit as st

from .vectorOps import calculate_vector

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def compute_similarity_vector(prev_pose, current_pose):
    """
    """
    try:
        prev_pose_left_shoulder             = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        prev_pose_left_elbow                = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        prev_pose_left_wrist                = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        prev_pose_v_left_shoulder_elbow     = calculate_vector(prev_pose_left_shoulder, prev_pose_left_elbow)
        prev_pose_v_left_elbow_wrist        = calculate_vector(prev_pose_left_elbow, prev_pose_left_wrist)

        prev_pose_right_shoulder            = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        prev_pose_right_elbow               = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        prev_pose_right_wrist               = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        prev_pose_v_right_shoulder_elbow    = calculate_vector(prev_pose_right_shoulder, prev_pose_right_elbow)
        prev_pose_v_right_elbow_wrist       = calculate_vector(prev_pose_right_elbow, prev_pose_right_wrist)
        
        current_pose_left_shoulder          = current_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        current_pose_left_elbow             = current_pose[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        current_pose_left_wrist             = current_pose[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        current_pose_v_left_shoulder_elbow  = calculate_vector(current_pose_left_shoulder, current_pose_left_elbow)
        current_pose_v_left_elbow_wrist     = calculate_vector(current_pose_left_elbow, current_pose_left_wrist)
        
        current_pose_right_shoulder         = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        current_pose_right_elbow            = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        current_pose_right_wrist            = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        current_pose_v_right_shoulder_elbow = calculate_vector(current_pose_right_shoulder, current_pose_right_elbow)
        current_pose_v_right_elbow_wrist    = calculate_vector(current_pose_right_elbow, current_pose_right_wrist)
        
        # Calculate cosine similarity for each pair of vectors
        similarity_left_shoulder_elbow      = cosine_similarity(prev_pose_v_left_shoulder_elbow, current_pose_v_left_shoulder_elbow)
        similarity_left_elbow_wrist         = cosine_similarity(prev_pose_v_left_elbow_wrist, current_pose_v_left_elbow_wrist)
        similarity_right_shoulder_elbow     = cosine_similarity(prev_pose_v_right_shoulder_elbow, current_pose_v_right_shoulder_elbow)
        similarity_right_elbow_wrist        = cosine_similarity(prev_pose_v_right_elbow_wrist, current_pose_v_right_elbow_wrist)
        
        # Average the similarities for a final similarity score
        similarity = (similarity_left_shoulder_elbow + similarity_left_elbow_wrist +
                      similarity_right_shoulder_elbow + similarity_right_elbow_wrist) / 4
        
        return similarity
    except Exception as e:
        st.write(e)        
        return 0

# def find_most_similar_keypoints_vectors(primary_frames, student_keypoints):
#     most_similar_keypoints = []
#     most_similar_keypoint_indices = []

#     for id, primary_frame in enumerate(primary_frames):
#         best_similarity = -1  # Initialize with a negative value
#         most_similar_keypoint = None
#         most_similar_keypoint_index = None

#         for idx, student_frame in enumerate(student_keypoints):
#             similarity = compute_similarity_vector(primary_frame, student_frame)
            
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 most_similar_keypoint = student_frame
#                 most_similar_keypoint_index = idx

#         most_similar_keypoints.append(most_similar_keypoint)
#         most_similar_keypoint_indices.append(most_similar_keypoint_index)

#     return most_similar_keypoints, most_similar_keypoint_indices

def find_most_similar_keypoints_vectors(primary_frames, student_keypoints):
    most_similar_keypoints = []
    most_similar_keypoint_indices = []
    last_selected_index = -1  # Initialize with -1 for the first iteration

    for id, primary_frame in enumerate(primary_frames):
        best_similarity = -1  # Initialize with a negative value
        most_similar_keypoint = None
        most_similar_keypoint_index = None

        # Start the inner loop from the next index after the last selected index
        for idx in range(last_selected_index + 1, len(student_keypoints)):
            student_frame = student_keypoints[idx]
            similarity = compute_similarity_vector(primary_frame, student_frame)
            
            if similarity > best_similarity:
                best_similarity = similarity
                most_similar_keypoint = student_frame
                most_similar_keypoint_index = idx

        most_similar_keypoints.append(most_similar_keypoint)
        most_similar_keypoint_indices.append(most_similar_keypoint_index)
        last_selected_index = most_similar_keypoint_index  # Update the last selected index

    return most_similar_keypoints, most_similar_keypoint_indices