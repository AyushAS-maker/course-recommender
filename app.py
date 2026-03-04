import streamlit as st
import pandas as pd
from model import recommend

df = pd.read_csv("coursera.csv")

st.title("AI Course Recommendation System")

course_list = df["course_title"].unique()

selected_course = st.selectbox("Choose a course", course_list)

if st.button("Recommend"):
    recommendations = recommend(selected_course)
    st.write("Recommended Courses:")
    for course in recommendations:
        st.write(course)