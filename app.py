import os
import copy
import tempfile

import streamlit as st

from pipelines import *
from styles import *
from constants import *

st.set_page_config(
    page_title="SPARTS Video tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("SPARTS Video tester")

# Sidebar with upload button
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov"]
)
selection = st.sidebar.radio("Select an option:", ("onlyHands", "allVectors"))

value_test_obj.setSliderValue()

col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("Instructor Video", expanded=False):
        st.video(ins_vid)

with col2:
    if uploaded_file is not None:
        with st.expander("Uploaded Video", expanded=False):
            st.video(uploaded_file)

if uploaded_file is not None and st.button("Process Video"):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        with st.expander("Logs", expanded=True):
            st.text("Step 0  : Uploaded video file successfully.")

            ins_keypoints, primary_frames = Step1(
                video_path=ins_vid,
                important_frames=important_frames,
            )
            st.text("Step 1  : Instaructor key points extracted successfully.")

            if selection == "onlyHands":
                stu_keypoints, most_similar_keypoints, most_similar_keypoint_indices = (
                    Step2(video_path=temp_file_path,
                          primary_frames=primary_frames, onlyHands=True)
                )
            else:
                stu_keypoints, most_similar_keypoints, most_similar_keypoint_indices = (
                    Step2(video_path=temp_file_path,
                          primary_frames=primary_frames, onlyHands=False)
                )

            st.text("Step 2.1: Student key points compared successfully.")
            st.text(
                f"Step 2.2: Extracted most similar keypoint indices: {most_similar_keypoint_indices}"
            )

            diff_list = Step3(
                primary_frames=primary_frames,
                most_similar_keypoint_indices=most_similar_keypoint_indices,
                student_keypoints=stu_keypoints,
            )
            diff_list2 = copy.deepcopy(diff_list)
            st.text("Step 3  : Angle difference Calculated.")

            response_list = Step4(diff_list, diff_list2)
            # response_list = ["sdd", "sds", "dds"]
            st.text("Step 4  : LLM text response gene.")

            stud_list, teach_list = Step5(
                temp_file_path,
                ins_vid,
                important_frames,
                most_similar_keypoint_indices,
                diff_list,
            )
            st.text("Step 5  : Image VIsualizations Generated.")

    if stud_list:
        Step6(stud_list, teach_list, response_list)

    for i, j in zip(stud_list, teach_list):
        if i  != "test.png":
            os.remove(i)
        if j  != "test.png":
            os.remove(j)
