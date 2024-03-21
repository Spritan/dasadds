import json
import yaml

from dotenv import dotenv_values
import google.generativeai as genai
import streamlit as st

from utils import cal_score, DROP_score, to_markdown

env_vars = dotenv_values()
        
genai.configure(api_key = st.secrets["API_KEY"])
model = genai.GenerativeModel("gemini-pro")


def Step4(diff_list: list, diff_list2: list) -> list[dict]:
    response_list = []
    for idx in range(len(diff_list)):
        try:
            response = model.generate_content(
                f"""actlike you are part of an evalution system for a sports academy where the posture is being checked between teacher and student\ 
                    while comapring body angles of instructors and students follwing the angle differences:
                    ###
                    {json.dumps(DROP_score(diff_list[idx]), indent=4)}
                    ###
                    summarise the result into human readable way. don't comment on any missing data and all just\ 
                    say what is readable from the provided data and make minor comments. 
                    Try to make the summary vivid with bullet points and any tips you might give the student.
                    
                    
                    Dont mention th exactly angle values in Findings make the language mopre lamen.
                    give the result is following yaml format://
                    ###
                    Findings:
                        - sdfdgdsgsd
                        - kjlkljlkjnkln n
                    Tips for Improvement:
                        - Keep the left arm relaxed and close to the body.
                        - Avoid raising the arm or bending it at an angle.
                    Comments:
                        - The student's posture is generally good, with the exception of the left arm position.
                        - Correcting this issue will help improve balance and reduce the risk of injury.
                        - Regular practice and attention to body position will help the student maintain proper posture during sports activities.

                    ###

                    """,
                stream=False,
            )
            print(response.candidates[0].content.parts[0].text)
            data = yaml.safe_load(response.candidates[0].content.parts[0].text)
            data['Score'] = f"{cal_score(diff_list2[idx])} out of {len(diff_list2[idx].keys())}"
        except:
            try:
                response = model.generate_content(
                    f"""actlike you are part of an evalution system for a sports academy where the posture is being checked between teacher and student\ 
                        while comapring body angles of instructors and students follwing the angle differences:
                        ###
                        {json.dumps(DROP_score(diff_list[idx]), indent=4)}
                        ###
                        summarise the result into human readable way. don't comment on any missing data and all just\ 
                        say what is readable from the provided data and make minor comments. 
                        Try to make the summary vivid with bullet points and any tips you might give the student.
                        
                        
                        Dont mention th exactly angle values in Findings make the language mopre lamen.
                        give the result is following yaml format://
                        ###
                        Findings:
                            - sdfdgdsgsd
                            - kjlkljlkjnkln n
                        Tips for Improvement:
                            - Keep the left arm relaxed and close to the body.
                            - Avoid raising the arm or bending it at an angle.
                        Comments:
                            - The student's posture is generally good, with the exception of the left arm position.
                            - Correcting this issue will help improve balance and reduce the risk of injury.
                            - Regular practice and attention to body position will help the student maintain proper posture during sports activities.

                        ###

                        """,
                    stream=False,
                )
                print(response.candidates[0].content.parts[0].text)
                data = yaml.safe_load(response.candidates[0].content.parts[0].text)
                data['Score'] = f"{cal_score(diff_list2[idx])} out of {len(diff_list2[idx].keys())}"
            except:
                response = model.generate_content(
                f"""actlike you are part of an evalution system for a sports academy where the posture is being checked between teacher and student\ 
                    while comapring body angles of instructors and students follwing the angle differences:
                    ###
                    {json.dumps(DROP_score(diff_list[idx]), indent=4)}
                    ###
                    summarise the result into human readable way. don't comment on any missing data and all just\ 
                    say what is readable from the provided data and make minor comments. 
                    Try to make the summary vivid with bullet points and any tips you might give the student.
                    
                    
                    Dont mention th exactly angle values in Findings make the language mopre lamen.
                    give the result is following yaml format://
                    ###
                    Findings:
                        - sdfdgdsgsd
                        - kjlkljlkjnkln n
                    Tips for Improvement:
                        - Keep the left arm relaxed and close to the body.
                        - Avoid raising the arm or bending it at an angle.
                    Comments:
                        - The student's posture is generally good, with the exception of the left arm position.
                        - Correcting this issue will help improve balance and reduce the risk of injury.
                        - Regular practice and attention to body position will help the student maintain proper posture during sports activities.

                    ###

                    """,
                stream=False,
                )
                print(response.candidates[0].content.parts[0].text)
                data = yaml.safe_load(response.candidates[0].content.parts[0].text)
                data['Score'] = f"{cal_score(diff_list2[idx])} out of {len(diff_list2[idx].keys())}"
            # print(data)
        response_list.append(data)
    return response_list
