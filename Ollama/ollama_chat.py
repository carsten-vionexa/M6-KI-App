# Ollama model via a streamlit interface

import streamlit as st
# Imports the Ollama Python client, used to interact with the Ollama language model API.
import ollama

st.title("Ollama!")
prompt = st.text_area(label = "Write your prompt.")
button = st.button("Okay")

# Handling User Input and Generating a Response
if button:
    if prompt:
        response = ollama.generate(model='llama3.1:latest', prompt=prompt) 
        st.write(response["response"])
        

# 1. User enters a prompt in the text area.

# 2. User clicks "Okay".

# 3. If the prompt is not empty, the app:

#     Sends the prompt to the Ollama model.
#     Receives a response.
#     Displays the model's response in the app.


