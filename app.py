import os
import copy
import tempfile

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, ClientSettings


from pipelines import *
from styles import *
from constants import *

import av
import cv2

import mediapipe as mp

mp_pose = mp.solutions.pose

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    rgb_frm = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(rgb_frm)
    # print(results)
    if results.pose_landmarks:
        mp.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    frm_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    cv2.putText(frm_bgr, "sdds", (int(frm_bgr.shape[1]/2), int(frm_bgr.shape[0]/2)), font, font_scale, (255, 255, 255), thickness=2)
    try:
        cv2.imwrite("/run/media/spritan/38c3181a-2d49-4ccb-bdbe-e934afa1eedc/test/test2.png", frm_bgr)
    except Exception as e:
        print("Error writing image:", e)
    return av.VideoFrame.from_ndarray(frm_bgr, format='bgr24')


st.set_page_config(
    page_title="SPARTS Video tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SPARTS Video tester")

st.sidebar.header("Options")

MODE_OPS = st.sidebar.radio("MODE", ["Upload", "Record"])

if MODE_OPS == "Upload":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov"]
    )
    selection = "allVectors"

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
                # print(important_frames)
                # print(primary_frames)
                if primary_frames[0]==0:
                    st.write("Step1")
                st.text("Step 1  : Instaructor key points extracted successfully.")

                if selection == "onlyHands":
                    (
                        stu_keypoints,
                        most_similar_keypoints,
                        most_similar_keypoint_indices,
                        cmnt_list,
                    ) = Step2(
                        video_path=temp_file_path,
                        primary_frames=primary_frames,
                        onlyHands=True,
                    )
                else:
                    (
                        stu_keypoints,
                        most_similar_keypoints,
                        most_similar_keypoint_indices,
                        cmnt_list,
                    ) = Step2(
                        video_path=temp_file_path,
                        primary_frames=primary_frames,
                        onlyHands=False,
                    )
                if primary_frames[0]==0:
                    st.write("Step2")
                st.text("Step 2.1: Student key points compared successfully.")
                st.text(
                    f"Step 2.2: Extracted most similar keypoint indices: {most_similar_keypoint_indices}"
                )
                # print("primary_frames", primary_frames)

                diff_list = Step3(
                    primary_frames=primary_frames,
                    most_similar_keypoint_indices=most_similar_keypoint_indices,
                    student_keypoints=stu_keypoints,
                )
                # st.write("stu_keypoints", len(stu_keypoints))
                # st.write("most_similar_keypoints", len(most_similar_keypoints))
                # st.write("most_similar_keypoint_indices", len(most_similar_keypoint_indices))
                if primary_frames[0]==0:
                    st.write("Step3")
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
            Step6(stud_list, teach_list, response_list, cmnt_list)

        for i, j in zip(stud_list, teach_list):
            if i != "test.png":
                os.remove(i)
            if j != "test.png":
                os.remove(j)


elif MODE_OPS == "Record":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(ins_vid)
            
    with col2:
        webrtc_streamer(
            key="key",
            video_frame_callback=video_frame_callback,
        )
