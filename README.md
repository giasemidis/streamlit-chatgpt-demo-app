# How to use this repo
1. Install the necessary packages, ideally within a conda environment, `pip install -r requirements.txt`
2. Get and Open AI key. If you have access to ChatGPT via the [Azure Studio](https://oai.azure.com/portal/921e49107f0843ce8bad20e6704c8e8c), go to "Playground", select either "Chat" or "Completions", and click on "View Code". A new pop-up window will show generated code. Copy the "Key" field at the bottom of the pop-up.
3. Create an `.env` file inside the `src` directory and paste:
```OPENAI_API_KEY = "<YOU KEY>"```

To run the MARTHA chatbot:
```
streamlit run src/main.py
```
The app will open in you browser window.

To run the "chat with your own data" chatbot:
```
streamlit run src/data_app.py
```

Due to bad design, do not run both apps at the same time.
