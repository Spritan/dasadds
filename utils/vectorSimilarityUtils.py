import numpy as np
import mediapipe as mp
from numpy.linalg import norm

import streamlit as st

from .vectorOps import calculate_vector
from constants import value_test_obj


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def compute_similarity_vector(prev_pose, current_pose, onlyHands=True):
    """
    """
    try:
        prev_pose_left_shoulder = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        prev_pose_left_elbow = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        prev_pose_left_wrist = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        prev_pose_v_left_shoulder_elbow = calculate_vector(
            prev_pose_left_shoulder, prev_pose_left_elbow)
        prev_pose_v_left_elbow_wrist = calculate_vector(
            prev_pose_left_elbow, prev_pose_left_wrist)

        prev_pose_left_hips = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        prev_pose_left_Knee = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
        prev_pose_left_ankle = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
        prev_pose_v_left_hips_Knee = calculate_vector(
            prev_pose_left_hips, prev_pose_left_Knee)
        prev_pose_v_left_Knee_ankle = calculate_vector(
            prev_pose_left_Knee, prev_pose_left_ankle)

        prev_pose_right_shoulder = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        prev_pose_right_elbow = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        prev_pose_right_wrist = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        prev_pose_v_right_shoulder_elbow = calculate_vector(
            prev_pose_right_shoulder, prev_pose_right_elbow)
        prev_pose_v_right_elbow_wrist = calculate_vector(
            prev_pose_right_elbow, prev_pose_right_wrist)

        prev_pose_right_hips = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        prev_pose_right_Knee = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
        prev_pose_right_ankle = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
        prev_pose_v_right_hips_Knee = calculate_vector(
            prev_pose_right_hips, prev_pose_right_Knee)
        prev_pose_v_right_Knee_ankle = calculate_vector(
            prev_pose_right_Knee, prev_pose_right_ankle)

        current_pose_left_shoulder = current_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        current_pose_left_elbow = current_pose[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        current_pose_left_wrist = current_pose[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        current_pose_v_left_shoulder_elbow = calculate_vector(
            current_pose_left_shoulder, current_pose_left_elbow)
        current_pose_v_left_elbow_wrist = calculate_vector(
            current_pose_left_elbow, current_pose_left_wrist)

        current_pose_left_hips = current_pose[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        current_pose_left_Knee = current_pose[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
        current_pose_left_ankle = current_pose[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
        current_pose_v_left_hips_Knee = calculate_vector(
            current_pose_left_hips, current_pose_left_Knee)
        current_pose_v_left_Knee_ankle = calculate_vector(
            current_pose_left_Knee, current_pose_left_ankle)

        current_pose_right_shoulder = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        current_pose_right_elbow = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        current_pose_right_wrist = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        current_pose_v_right_shoulder_elbow = calculate_vector(
            current_pose_right_shoulder, current_pose_right_elbow)
        current_pose_v_right_elbow_wrist = calculate_vector(
            current_pose_right_elbow, current_pose_right_wrist)

        current_pose_right_hips = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        current_pose_right_Knee = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
        current_pose_right_ankle = current_pose[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
        current_pose_v_right_hips_Knee = calculate_vector(
            current_pose_right_hips, current_pose_right_Knee)
        current_pose_v_right_Knee_ankle = calculate_vector(
            current_pose_right_Knee, current_pose_right_ankle)

        # Calculate cosine similarity for each pair of vectors
        similarity_left_shoulder_elbow = cosine_similarity(
            prev_pose_v_left_shoulder_elbow, current_pose_v_left_shoulder_elbow)
        similarity_left_elbow_wrist = cosine_similarity(
            prev_pose_v_left_elbow_wrist, current_pose_v_left_elbow_wrist)
        similarity_left_hips_Knee = cosine_similarity(
            prev_pose_v_left_hips_Knee, current_pose_v_left_hips_Knee)
        similarity_left_Knee_ankle = cosine_similarity(
            prev_pose_v_left_Knee_ankle, current_pose_v_left_Knee_ankle)
        similarity_right_shoulder_elbow = cosine_similarity(
            prev_pose_v_right_shoulder_elbow, current_pose_v_right_shoulder_elbow)
        similarity_right_elbow_wrist = cosine_similarity(
            prev_pose_v_right_elbow_wrist, current_pose_v_right_elbow_wrist)
        similarity_right_hips_Knee = cosine_similarity(
            prev_pose_v_right_hips_Knee, current_pose_v_right_hips_Knee)
        similarity_right_Knee_ankle = cosine_similarity(
            prev_pose_v_right_Knee_ankle, current_pose_v_right_Knee_ankle)

        if onlyHands:
            # Average the similarities for a final similarity score
            similarity = (similarity_left_shoulder_elbow + similarity_left_elbow_wrist +
                          similarity_right_shoulder_elbow + similarity_right_elbow_wrist) / 4

        else:
            similarity = (
                similarity_left_shoulder_elbow * 2 + 
                similarity_left_elbow_wrist * 2 +
                similarity_right_shoulder_elbow * 2 + 
                similarity_right_elbow_wrist * 2 +
                similarity_left_hips_Knee + 
                similarity_left_Knee_ankle + 
                similarity_right_hips_Knee +
                similarity_right_Knee_ankle
            ) / 12

        return similarity
    except Exception as e:
        st.write(e)
        return 0


def find_most_similar_keypoints_vectors(primary_frames, student_keypoints, onlyHands):
    most_similar_keypoints = []
    most_similar_keypoint_indices = []
    last_selected_index = -1  # Initialize with -1 for the first iteration

    for id, primary_frame in enumerate(primary_frames):
        best_similarity = -1  # Initialize with a negative value
        most_similar_keypoint = None
        most_similar_keypoint_index = 0

        for idx in range(last_selected_index, len(student_keypoints)):
            student_frame = student_keypoints[idx]
            
            similarity = compute_similarity_vector(
                primary_frame, student_frame, onlyHands
            )

            similarity_threshold = value_test_obj.getSliderValue()

            # st.write(similarity_threshold)

            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                most_similar_keypoint = student_frame
                most_similar_keypoint_index = idx

            elif similarity > best_similarity and similarity <= similarity_threshold:
                best_similarity = similarity
                most_similar_keypoint = student_frame
                most_similar_keypoint_index = 0

        if most_similar_keypoint is not None:
            most_similar_keypoints.append(most_similar_keypoint)
            most_similar_keypoint_indices.append(most_similar_keypoint_index)
            last_selected_index = most_similar_keypoint_index  # Update the last selected index

    return most_similar_keypoints, most_similar_keypoint_indices

