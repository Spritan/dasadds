import streamlit as st

ins_vid = "./data/punchInstuctorVideo.mp4"
# important_frames = [0, 87, 125, 248]
important_frames = [0, 125, 248]

class value_test():
    def __init__(self):
        self.slider_value = 0.5

    def setSliderValue(self):
        self.slider_value = st.sidebar.slider('Select similarity threshold:', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    def getSliderValue(self):
        return self.slider_value

value_test_obj = value_test()