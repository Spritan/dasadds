# import numpy as np
from utils import extract_keypoints_from_video

def readKeypoints(video_path:str):
    instructor_keypoints=extract_keypoints_from_video(video_path=video_path)
    return instructor_keypoints
    
def Step1(video_path:str, important_frames:list[int])-> tuple[list,list]:
    keypoints = readKeypoints(video_path)
    primary_frames = [keypoints[i] for i in important_frames]
    return keypoints, primary_frames
    