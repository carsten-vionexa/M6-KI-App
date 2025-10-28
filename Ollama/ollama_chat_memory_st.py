# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:29:15 2025

@author: milos
"""

import streamlit as st
import ollama

MODEL = 'llama3.1'

# Initialize conversation history in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title("Llama 3.1 Chatbot with Ollama and Streamlit")

# Display previous messages
for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input box at the bottom
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Get model response
    response = ollama.chat(model=MODEL, messages=st.session_state['messages'])
    assistant_reply = response['message']['content']

    # Add assistant message to history
    st.session_state['messages'].append({'role': 'assistant', 'content': assistant_reply})
    with st.chat_message('assistant'):
        st.markdown(assistant_reply)
