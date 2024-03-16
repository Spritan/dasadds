import streamlit as st

def Step6(stud_list,teach_list,response_list_txt):
    
    for i,j,k in zip(stud_list,teach_list,response_list_txt):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Student Image")
            st.image(i, use_column_width=True)

        with col2:
            st.header("Instructor Image")
            st.image(j, use_column_width=True)

        with col3:
            st.header("Feedback")
            st.markdown(k)
    pass