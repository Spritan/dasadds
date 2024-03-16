import math
import mediapipe as mp

def calculate_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

def calculate_angle(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in v2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    return math.degrees(math.acos(cosine_angle))

def angle_diff(angles1:dict, angles2:dict)->dict:
    angle_differences = {}

    for key in angles1.keys():
        angle_differences[key] = angles1[key] - angles2[key]
    return angle_differences

def middle_keypoint(keypoints1, keypoints2):
    keypoint = ((keypoints1[0]+keypoints2[0])/2,
                (keypoints1[1]+keypoints2[1])/2,
                (keypoints1[2]+keypoints2[2])/2)
    return keypoint

def calculate_angles(keypoints):
    angles = {}

    # Calculate angles for left arm
    left_shoulder = keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = keypoints[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = keypoints[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
    v_left_shoulder_elbow = calculate_vector(left_shoulder, left_elbow)
    v_left_elbow_wrist = calculate_vector(left_elbow, left_wrist)

    # Calculate angles for right arm
    right_shoulder = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
    v_right_shoulder_elbow = calculate_vector(right_shoulder, right_elbow)
    v_right_elbow_wrist = calculate_vector(right_elbow, right_wrist)

    # Calculate angles for left leg
    left_hip = keypoints[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    left_knee = keypoints[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = keypoints[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    v_left_hip_knee = calculate_vector(left_hip, left_knee)
    v_left_knee_ankle = calculate_vector(left_knee, left_ankle)

    # Calculate angles for right leg
    right_hip = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = keypoints[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
    v_right_hip_knee = calculate_vector(right_hip, right_knee)
    v_right_knee_ankle = calculate_vector(right_knee, right_ankle)
    
    v_left_shoulder_hip = calculate_vector(left_shoulder, left_hip)
    
    v_right_shoulder_hip = calculate_vector(right_shoulder, right_hip)
    
    neck_keypoint = middle_keypoint(
        keypoints[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
        keypoints[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
    
    v_right_shoulder_neck = calculate_vector(right_shoulder, neck_keypoint)
    v_right_nose_neck = calculate_vector(
        keypoints[mp.solutions.pose.PoseLandmark.NOSE.value], 
        neck_keypoint
        )
    
    # v_right_shoulder_hip = calculate_vector(right_shoulder, right_elbow)

    angles['face_to_shoulder_right'] =  calculate_angle(v_right_shoulder_neck, v_right_nose_neck)
    angles['left_arm_angle'] = calculate_angle(v_left_shoulder_elbow, v_left_elbow_wrist)
    angles['right_arm_angle'] = calculate_angle(v_right_shoulder_elbow, v_right_elbow_wrist)
    # angles['left_elbow_hip'] = calculate_angle(v_left_shoulder_hip, v_left_shoulder_elbow)
    # angles['right_elbow_hip'] = calculate_angle(v_right_shoulder_hip, v_right_shoulder_elbow)
    angles['left_leg_angle'] = calculate_angle(v_left_hip_knee, v_left_knee_ankle)
    angles['right_leg_angle'] = calculate_angle(v_right_hip_knee, v_right_knee_ankle)
    angles['left_hip_angle'] = calculate_angle(v_left_hip_knee, v_left_shoulder_hip)
    angles['right_hip_angle'] = calculate_angle(v_right_hip_knee, v_right_shoulder_hip)

    return angles