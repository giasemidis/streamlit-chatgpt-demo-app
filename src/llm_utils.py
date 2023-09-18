import os
import re
import openai

from dotenv import load_dotenv, find_dotenv
import streamlit as st

if os.environ.get('OPENAI_API_KEY') is not None:
    openai.api_key = os.environ['OPENAI_API_KEY']
else:
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

openai.api_type = "azure"
openai.api_base = "https://wppendlz07-dev-openai-002.openai.azure.com/"


def chat_api(
    messages, engine="GPT-4", temperature=0.0, max_tokens=3_000, top_p=0.5
):
    """
    The chat API endpoint of the ChatGPT

    Args:
        messages (str): The input messages to the chat API
        engine (str): The engine, i.e. the LLM
        temperature (float): The temperature parameter
        max_tokens (int): Max number of tokens parameters
        top_p (float): Top P parameter

    Returns:
        str: The LLM response
    """
    plot_flag = False
    openai.api_version = "2023-03-15-preview"

    if "plot" in messages[-1]["content"].lower():
        plot_flag = True
        code_prompt = """
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        messages.append({
            "role": "assistant",
            "content": code_prompt
        })

    response = openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    if plot_flag:
        code = extract_python_code(
            response["choices"][0]["message"]["content"])
        code = code.replace("fig.show()", "")
        code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
        st.write(f"```{code}")
        exec(code)

    return response["choices"][0]["message"]


def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]
