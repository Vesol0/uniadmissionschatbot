import time
from agent import get_ai_response
import streamlit as st
from save_to_csv import save_to_csv

st.title("Admissions Chatbot")
st.download_button(
    label="Download History",
    data=save_to_csv(st.session_state.history),
    file_name="history.csv",
    mime="text/csv",
    icon=":material/download:"
)
def chat_stream(prompt):
    response = f'You said, "{prompt}" ...interesting.'
    for char in response:
        yield char
        time.sleep(.02)
if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])





if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("...", show_time=True):
            response = get_ai_response(prompt)
            st.write(response)
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                args=[len(st.session_state.history)],
            )
    st.session_state.history.append({"role": "assistant", "content":response})













