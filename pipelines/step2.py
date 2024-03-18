import streamlit as st

from utils import find_most_similar_keypoints_vectors, extract_keypoints_from_video

def readKeypoints(video_path:str):
    instructor_keypoints=extract_keypoints_from_video(video_path=video_path)
    
    return instructor_keypoints
    
def Step2(video_path:str, primary_frames:list, onlyHands=True)-> tuple[list,list,list]:
    keypoints = readKeypoints(video_path)
    most_similar_keypoints, most_similar_keypoint_indices, cmnt_list = find_most_similar_keypoints_vectors(primary_frames, keypoints, onlyHands=onlyHands)
    
    return keypoints, most_similar_keypoints, most_similar_keypoint_indices, cmnt_list