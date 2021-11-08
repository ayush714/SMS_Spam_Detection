import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from main import predict

st.write(
    """ 

# SMS Spam Detection Application 

You can use this tool, given an sms message, to detect if it is a spam or not.  
"""
)

message = st.text_area("Enter a message to be classified")

if st.button("Classify"):
    result = predict(message)
    if result == 1:
        st.success("The message is a spam")
    else:
        st.success("The message is ham")
