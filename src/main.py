import streamlit as st
from streamlit_chat import message
from llm_utils import chat_api


def get_text():
    """Input text by the user"""
    input_text = st.text_input(
        "Ask me your question.",
        "", key="input"
    )
    return input_text


def sidebar():
    """App sidebar content"""

    model_help = (
        "The available models for the selected API endpoint. Same prompt might "
        "return different results for different models. Epxerimentation is "
        "recommended."
    )
    model = st.selectbox(
        label="Available Models",
        options=["gpt-4-32k", "GPT-4", "gpt-35-turbo"],
        help=model_help
    )

    temperature = st.slider(
        label="Temperature",
        value=0.0,
        min_value=0.,
        max_value=1.,
        step=0.01,
        help=(
            "Controls randomness. Lowering the temperature means that the model "
            "will produce more repetitive and deterministic responses. Increasing "
            "the temperature will result in more unexpected or creative responses. "
            "Try adjusting temperature or Top P but not both."
        )
    )
    max_tokens = st.slider(
        label="Max length (tokens)",
        value=3_000,
        min_value=0,
        max_value=8_192,
        step=1,
        help=(
            "Set a limit on the number of tokens per model response. The API "
            "supports a maximum of 8192 tokens shared between the prompt "
            "(including system message, examples, message history, and user query) "
            "and the model's response. One token is roughly 4 characters for "
            "typical English text."
        )

    )
    top_p = st.slider(
        label="Top P",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "Similar to temperature, this controls randomness but uses a different "
            "method. Lowering Top P will narrow the model’s token selection to "
            "likelier tokens. Increasing Top P will let the model choose from "
            "tokens with both high and low likelihood. Try adjusting temperature "
            "or Top P but not both."
        )
    )
    out_dict = {
        "engine": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    return out_dict


def chatbot():
    """
    Main chatbox function based on ChatCompletion API
    """

    st.title("MARTHA: Your marketing and advertising expert chatbox.")

    with st.sidebar:
        model_params = sidebar()

    greeting_bot_msg = (
        "Hi, I am MARTHA, your marketing & advertising expert. "
        "Ask me any related question.\n"
        "Ah! I have no knowledge of 2022 onwards, because I am powered by "
        "ChatGPT. So, I don't do predictions.\n"
        "*Example*: 'What are the implications of the death of third party "
        "cookies for the industry?'\n"
        "I don't answer questions like 'Who was US president in 2010?'"
    )

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = [greeting_bot_msg]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    prompt = (
        "Classify if the following prompt questions are related to marketing "
        "and advertising. If they are, answer the question. If they are not, "
        "reply only 'This is not a marketing question' and do not answer the "
        "question. Allow any questions about plots."
    )
    # prompt = "Answer user's questions"
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": prompt},
        ]

    user_input = get_text()

    if user_input:
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        response = chat_api(st.session_state["messages"], **model_params)
        st.session_state.past.append(user_input)
        if response is not None:
            st.session_state.generated.append(response["content"])
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
    chatbot()
