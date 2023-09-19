import streamlit as st
from streamlit_chat import message
import pandas as pd
from main import get_text
from llm_utils import chat_with_data_api


def chat_with_data():

    uploaded_file = st.file_uploader(
        label="Choose file",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.write("Please upload the a file first.")

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Please upload your data"]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = get_text()

    if user_input:
        st.session_state["messages"].append(user_input)
        response = chat_with_data_api(
            df, user_input, chat_history=st.session_state["generated"])
        st.session_state.past.append(user_input)
        if response is not None:
            st.session_state.generated.append(response)
            st.session_state["messages"].append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            if i - 1 >= 0:
                message(
                    st.session_state["past"][i - 1],
                    is_user=True,
                    key=str(i) + "_user"
                )


if __name__ == "__main__":
    chat_with_data()
