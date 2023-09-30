import streamlit as st
from streamlit_chat import message
import pandas as pd
from main import get_text, sidebar
from llm_utils import chat_with_data_api


def chat_with_data():

    st.title("Chat with, query and plot your own data")

    with st.sidebar:
        model_params = sidebar()

    uploaded_file = st.file_uploader(
        label="Choose file",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        prompt = f"""You are a python expert. You will be given questions for
            manipulating an input dataframe.
            If the question asks for a particular country
            The available columns are: `{df.columns}`.
            Use them for extracting the relevant data.

        """
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "system", "content": prompt}]
    else:
        st.write("Please upload a csv file")
        st.session_state["messages"] = []
        df = pd.DataFrame([])

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Please upload your data"]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if ((len(st.session_state["past"]) > 0)
            and (user_input == st.session_state["past"][-1])):
        user_input = ""

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        if df.empty:
            st.warning("Dataframe is empty, upload a valid file", icon="⚠️")
        else:
            response = chat_with_data_api(df, **model_params)
            st.session_state.past.append(user_input)
            if response is not None:
                st.session_state.generated.append(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response})

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
