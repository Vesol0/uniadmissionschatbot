import streamlit as st
import pandas as pd

def save_to_csv(messages):
    if messages is not None:
       return pd.DataFrame.from_dict(messages).to_csv()
    else:
        st.warning("Cannot save.")
