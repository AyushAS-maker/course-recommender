import pandas as pd
import streamlit as st

df = pd.read_csv("coursera.csv")

st.write("Columns in dataset:")
st.write(df.columns)

st.stop()