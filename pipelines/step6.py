import streamlit as st

def Step6(stud_list,teach_list,response_list_txt, cmnt_list):
    
    for i,j,k,l in zip(stud_list,teach_list,response_list_txt, cmnt_list):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Student Image")
            st.image(i, use_column_width=True)

        with col2:
            st.header("Instructor Image")
            st.image(j, use_column_width=True)

        with col3:
            st.header("Feedback")
            if i=="test.png":
                st.markdown(l)
            else:
                st.markdown(l)
                st.markdown(k)
    pass