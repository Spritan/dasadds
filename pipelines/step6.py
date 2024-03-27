import streamlit as st

def Step6(
    stud_list:list,
    teach_list:list,
    response_list_txt:list, 
    cmnt_list:list
    )->None:
    # st.write("stud_list",len(stud_list))
    # st.write("teach_list",len(teach_list))
    # st.write("response_list_txt",len(response_list_txt))
    # st.write("cmnt_list",len(cmnt_list))
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
                st.markdown(f"# Score: {k['Score']}")
                st.markdown("## Tips for Improvement:")
                try:
                    for items in k['Tips for Improvement']:
                        st.markdown(f" - ##### {items}")
                except:
                    st.markdown(" - ##### No comments.")

                st.markdown("### Findings:")
                try:
                    for items in k['Findings']:
                        st.markdown(f" - {items}")
                except:
                    st.markdown(" - No comments.")
                st.markdown("### Comments:")
                try:
                    for items in k['Comments']:
                        st.markdown(f" - {items}")
                except:
                    st.markdown(" - No comments.")
                if l!="":
                    st.markdown(f" - {l}.")
