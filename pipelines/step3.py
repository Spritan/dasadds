from utils import calculate_angles, angle_diff

def Step3(primary_frames:list, most_similar_keypoint_indices:list, student_keypoints:list)->list:
    diff_list=[]
    for i, j in zip(primary_frames, most_similar_keypoint_indices):
        if (i is not None) and (j is not None):
            a = calculate_angles(i)
            b = calculate_angles(student_keypoints[int(j)])
            z = angle_diff(a,b)
            diff_list.append(z)
    return diff_list