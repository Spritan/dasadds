import streamlit as st

from utils import draw_right_arm_new2

def Step5(
    student_vid: str,
    teacher_vid: str,
    important_frames: list,
    most_similar_keypoint_indices: list,
    diff_list: list,
) -> tuple[list, list]:
    stud_list = []
    teach_list = []
    for idx, tup in enumerate(
        zip(important_frames, most_similar_keypoint_indices, diff_list)
    ):
        i, j, dif_ang = tup
        # st.write(f"{idx}, {j}")
        if idx == 0 or j > 0:
            ang_lst = []
            for keys, values in dif_ang.items():
                if values >= 25 and keys not in [
                    "face_to_shoulder_right",
                    "right_elbow_hip",
                    "left_elbow_hip",
                ]:
                    ang_lst.append(keys)
            draw_right_arm_new2(student_vid, j, "student_", ang_lst)
            stud_list.append(f"student__{j}.jpg")
            # print(f"student__{j}.jpg")
            draw_right_arm_new2(teacher_vid, i, "teacher_", ang_lst)
            teach_list.append(f"teacher__{i}.jpg")
        else:
            ang_lst = []
            for keys, values in dif_ang.items():
                if values >= 25 and keys not in [
                    "face_to_shoulder_right",
                    "right_elbow_hip",
                    "left_elbow_hip",
                ]:
                    ang_lst.append(keys)
            # draw_right_arm_new2(student_vid, j, "student_", ang_lst)
            stud_list.append(f"test.png")
            draw_right_arm_new2(teacher_vid, i, "teacher_", ang_lst)
            teach_list.append(f"teacher__{i}.jpg")
    
    return stud_list, teach_list
