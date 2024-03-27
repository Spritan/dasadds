# import numpy as np
import streamlit as st
from utils import extract_keypoints_from_video

def readKeypoints(video_path:str):
    instructor_keypoints=extract_keypoints_from_video(video_path=video_path)
    return instructor_keypoints
    
def Step1(video_path:str, important_frames:list[int])-> tuple[list,list]:
    keypoints = readKeypoints(video_path)
    primary_frames = [keypoints[i] for i in important_frames]
    for idx, j in enumerate(zip(important_frames, primary_frames)):
        imp_frame, p_frame = j
        if p_frame == None:
            primary_frames[idx] = keypoints[idx+1]
    print("primary_frames", primary_frames)
    return keypoints, primary_frames
    