import streamlit as st

from utils import calculate_angles, angle_diff

def Step3(
    primary_frames: list, most_similar_keypoint_indices: list, student_keypoints: list
) -> list:
    diff_list = []
    for idx, z in enumerate(zip(primary_frames, most_similar_keypoint_indices)):
        i, j = z
        if (i is not None) and (j is not None):
            # print(i)
            a = calculate_angles(i)
            b = calculate_angles(student_keypoints[int(j)])
            z = angle_diff(a, b)
            # st.write(z)
            diff_list.append(z)
        else:
            # st.write(idx)
            # st.write("i", i)
            # st.write("j",j)
            pass
    return diff_list


