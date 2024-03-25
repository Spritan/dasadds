import math
import numpy as np
import mediapipe as mp
from numpy.linalg import norm

import streamlit as st

from .vectorOps import calculate_vector
from constants import value_test_obj

def cosine_similarity(
    v1:tuple[int,int,int], 
    v2:tuple[int,int,int]
    )->float:
    """_summary_

    Args:
        v1 (tuple[int,int,int]): _description_
        v2 (tuple[int,int,int]): _description_

    Returns:
        float: _description_
    """
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def compute_similarity_vector(
    prev_pose:list,
    current_pose:list, 
    onlyHands=True
    )->float:
    """_summary_

    Args:
        prev_pose (list): _description_
        current_pose (list): _description_
        onlyHands (bool, optional): _description_. Defaults to True.

    Returns:
        float: _description_
    """
    try:
        prev_pose_left_shoulder = prev_pose[
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        ]
        prev_pose_left_elbow = prev_pose[
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
        ]
        prev_pose_left_wrist = prev_pose[
            mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        ]
        prev_pose_v_left_shoulder_elbow = calculate_vector(
            prev_pose_left_shoulder, prev_pose_left_elbow
        )
        prev_pose_v_left_elbow_wrist = calculate_vector(
            prev_pose_left_elbow, prev_pose_left_wrist
        )

        prev_pose_left_hips = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        prev_pose_left_Knee = prev_pose[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
        prev_pose_left_ankle = prev_pose[
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
        ]
        prev_pose_v_left_hips_Knee = calculate_vector(
            prev_pose_left_hips, prev_pose_left_Knee
        )
        prev_pose_v_left_Knee_ankle = calculate_vector(
            prev_pose_left_Knee, prev_pose_left_ankle
        )

        prev_pose_right_shoulder = prev_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        ]
        prev_pose_right_elbow = prev_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        ]
        prev_pose_right_wrist = prev_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        ]
        prev_pose_v_right_shoulder_elbow = calculate_vector(
            prev_pose_right_shoulder, prev_pose_right_elbow
        )
        prev_pose_v_right_elbow_wrist = calculate_vector(
            prev_pose_right_elbow, prev_pose_right_wrist
        )

        prev_pose_right_hips = prev_pose[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        prev_pose_right_Knee = prev_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
        ]
        prev_pose_right_ankle = prev_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        prev_pose_v_right_hips_Knee = calculate_vector(
            prev_pose_right_hips, prev_pose_right_Knee
        )
        prev_pose_v_right_Knee_ankle = calculate_vector(
            prev_pose_right_Knee, prev_pose_right_ankle
        )

        current_pose_left_shoulder = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        ]
        current_pose_left_elbow = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
        ]
        current_pose_left_wrist = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        ]
        current_pose_v_left_shoulder_elbow = calculate_vector(
            current_pose_left_shoulder, current_pose_left_elbow
        )
        current_pose_v_left_elbow_wrist = calculate_vector(
            current_pose_left_elbow, current_pose_left_wrist
        )

        current_pose_left_hips = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        ]
        current_pose_left_Knee = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
        ]
        current_pose_left_ankle = current_pose[
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
        ]
        current_pose_v_left_hips_Knee = calculate_vector(
            current_pose_left_hips, current_pose_left_Knee
        )
        current_pose_v_left_Knee_ankle = calculate_vector(
            current_pose_left_Knee, current_pose_left_ankle
        )

        current_pose_right_shoulder = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        ]
        current_pose_right_elbow = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        ]
        current_pose_right_wrist = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        ]
        current_pose_v_right_shoulder_elbow = calculate_vector(
            current_pose_right_shoulder, current_pose_right_elbow
        )
        current_pose_v_right_elbow_wrist = calculate_vector(
            current_pose_right_elbow, current_pose_right_wrist
        )

        current_pose_right_hips = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
        ]
        current_pose_right_Knee = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
        ]
        current_pose_right_ankle = current_pose[
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        current_pose_v_right_hips_Knee = calculate_vector(
            current_pose_right_hips, current_pose_right_Knee
        )
        current_pose_v_right_Knee_ankle = calculate_vector(
            current_pose_right_Knee, current_pose_right_ankle
        )

        # Calculate cosine similarity for each pair of vectors
        similarity_left_shoulder_elbow = cosine_similarity(
            prev_pose_v_left_shoulder_elbow, current_pose_v_left_shoulder_elbow
        )
        similarity_left_elbow_wrist = cosine_similarity(
            prev_pose_v_left_elbow_wrist, current_pose_v_left_elbow_wrist
        )
        similarity_left_hips_Knee = cosine_similarity(
            prev_pose_v_left_hips_Knee, current_pose_v_left_hips_Knee
        )
        similarity_left_Knee_ankle = cosine_similarity(
            prev_pose_v_left_Knee_ankle, current_pose_v_left_Knee_ankle
        )
        similarity_right_shoulder_elbow = cosine_similarity(
            prev_pose_v_right_shoulder_elbow, current_pose_v_right_shoulder_elbow
        )
        similarity_right_elbow_wrist = cosine_similarity(
            prev_pose_v_right_elbow_wrist, current_pose_v_right_elbow_wrist
        )
        similarity_right_hips_Knee = cosine_similarity(
            prev_pose_v_right_hips_Knee, current_pose_v_right_hips_Knee
        )
        similarity_right_Knee_ankle = cosine_similarity(
            prev_pose_v_right_Knee_ankle, current_pose_v_right_Knee_ankle
        )

        if onlyHands:
            # Average the similarities for a final similarity score
            similarity = (
                similarity_left_shoulder_elbow
                + similarity_left_elbow_wrist
                + similarity_right_shoulder_elbow
                + similarity_right_elbow_wrist
            ) / 4

        else:
            similarity = (
                similarity_left_shoulder_elbow * 2
                + similarity_left_elbow_wrist * 2
                + similarity_right_shoulder_elbow * 2
                + similarity_right_elbow_wrist * 2
                + similarity_left_hips_Knee
                + similarity_left_Knee_ankle
                + similarity_right_hips_Knee
                + similarity_right_Knee_ankle
            ) / 12

        return similarity
    except Exception as e:
        print(e)
        return 0

def find_most_similar_keypoints_vectors(
    primary_frames: list, 
    student_keypoints: list, 
    onlyHands: bool
    )->tuple[list,list,list]:
    """_summary_

    Args:
        primary_frames (list): _description_
        student_keypoints (list): _description_
        onlyHands (bool): _description_

    Returns:
        tuple[list,list,list]: _description_
    """
    most_similar_keypoints = []
    most_similar_keypoint_indices = []
    cmnt_list = []
    last_selected_index = -1

    tot_student_frames = len(student_keypoints)
    tot_frames = len(primary_frames)

    frame_ranges = []
    buffer_factor = 0.13

    for ink in range(tot_frames):
        if ink == 0:
            frame_ranges.append(
                (
                    0,
                    math.ceil(
                        ((ink + 1) * tot_student_frames / tot_frames)
                        # + (buffer_factor * tot_student_frames)
                    ),
                )
            )
        elif ink == tot_frames - 1:
            frame_ranges.append(
                (
                    math.ceil(
                        ((ink) * tot_student_frames / tot_frames)
                        # - (buffer_factor * tot_student_frames)
                    ),
                    tot_student_frames - 1,
                )
            )
        else:
            frame_ranges.append(
                (
                    math.ceil(
                        ((ink) * tot_student_frames / tot_frames)
                        # - (buffer_factor * tot_student_frames)
                    ),
                    math.ceil(
                        ((ink + 1) * tot_student_frames / tot_frames)
                        # + (buffer_factor * tot_student_frames)
                    ),
                )
            )

    # st.write(frame_ranges)
    for id, primary_frame in enumerate(primary_frames):
        
        best_similarity = -1
        most_similar_keypoint = None
        most_similar_keypoint_index = 0
        cmnt = ""

        start, end = frame_ranges[id]
        print(start, end)

        for idx in range(start, end):
            # st.write(idx)
            student_frame = student_keypoints[idx]
            # st.write("default")
            similarity = compute_similarity_vector(
                primary_frame, student_frame, onlyHands
            )
            print("start, end", similarity)
            similarity_threshold = value_test_obj.getSliderValue()
            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                most_similar_keypoint = student_frame
                most_similar_keypoint_index = idx
                cmnt = ""

            elif similarity > best_similarity and similarity <= similarity_threshold:
                best_similarity = similarity
                most_similar_keypoint = student_frame
                most_similar_keypoint_index = 0
                cmnt = "### No proper pose executed or you are not following fproper timings and rythem compared to the Instructor, try again..."

        if start > 0 and most_similar_keypoint_index == 0:
            best_similarity = -1
            most_similar_keypoint = None
            most_similar_keypoint_index = 0
            for idx in range(0, tot_student_frames):
                student_frame = student_keypoints[idx]
                # st.write("start > 0 and most_similar_keypoint_index == 0:", idx)
                similarity = compute_similarity_vector(
                    primary_frame, student_frame, onlyHands
                )
                print("start > 0 and most_similar_keypoint_index", similarity)
                similarity_threshold = value_test_obj.getSliderValue()
                if similarity > best_similarity and similarity > similarity_threshold:
                    best_similarity = similarity
                    most_similar_keypoint = student_frame
                    most_similar_keypoint_index = idx
                    cmnt = "### No proper pose executed or you are not following fproper timings but your most similar pose pose is ..."

        if start == 0 and most_similar_keypoint_index == 0:
            cmnt = ""

        if most_similar_keypoint is not None:
            most_similar_keypoints.append(most_similar_keypoint)
            most_similar_keypoint_indices.append(most_similar_keypoint_index)
            last_selected_index = most_similar_keypoint_index
            cmnt_list.append(cmnt)

    return most_similar_keypoints, most_similar_keypoint_indices, cmnt_list
