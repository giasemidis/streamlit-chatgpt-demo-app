# How to use this repo
1. Install the necessary packages, ideally within a conda environment, `pip install -r requirements.txt`
2. Get an Open AI key. Create one on your [ChatGPT API account](https://platform.openai.com/account/api-keys)
3. Create an `.env` file inside the `src` directory and paste:
```OPENAI_API_KEY = "<YOUR KEY>"```

To run the chatbot:
```
streamlit run src/main.py
```
The app will open in your browser window.

To run the "chat with your own data" chatbot:
```
streamlit run src/data_app.py
```

Due to bad design, do not run both apps at the same time.
