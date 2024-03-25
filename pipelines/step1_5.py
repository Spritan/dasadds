import av
import cv2

import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase

mp_pose = mp.solutions.pose.Pose()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0
        
    def recv(self, frame):
        # img = frame.to_ndarray(format="bgr24")
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # i =self.i+1
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
        #     cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
        #     cv2.putText(img, 'F-' + str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        frm = frame.to_ndarray(format="bgr24")
        rgb_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb_frm)
        if results.pose_landmarks:
            print(results.pose_landmarks)
        if results.pose_landmarks:
            mp.drawing_utils.draw_landmarks(frm, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        frm_bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        cv2.putText(frm_bgr, "sdds", 
                    (int(frm_bgr.shape[1]/2), int(frm_bgr.shape[1]/2)), font, font_scale, (255, 255, 255), thickness=2)
        
        return frm_bgr
