from utils import draw_right_arm_new2


def Step5(student_vid:str, teacher_vid:str, important_frames:list, most_similar_keypoint_indices:list, diff_list:list)-> tuple[list, list]:
    stud_list = []
    teach_list = []
    for i,j, dif_ang in zip(important_frames, most_similar_keypoint_indices, diff_list):
        ang_lst = []
        for keys, values in dif_ang.items():
            if values >=25 and keys not in ['face_to_shoulder_right','right_elbow_hip','left_elbow_hip']:
                ang_lst.append(keys)
        draw_right_arm_new2(student_vid,j, "student_",ang_lst)
        stud_list.append(f"student__{j}.jpg")
        draw_right_arm_new2(teacher_vid,i, "teacher_",ang_lst)
        teach_list.append(f"teacher__{i}.jpg")
    return stud_list, teach_list