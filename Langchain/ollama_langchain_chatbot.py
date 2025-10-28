# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 19:56:02 2025

@author: milos
"""

"""
LangChain Chatbot with Ollama Local Model
-----------------------------------------

This script demonstrates how to build a stateful chatbot using LangChain and an Ollama local LLM.
It covers setup, conversation memory, prompt templates, conversation history management,
and streaming responses.

"""

# 1. Setup and Imports
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages

# 2. Initialize the Ollama Model
model = ChatOllama(model="llama3.1")

#%% 3. Stateless Chat Example

# The model alone does not remember previous messages
stateless_response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print(stateless_response.content)

followup_response = model.invoke([HumanMessage(content="What's my name?")])
print(followup_response.content)

#%% 4. Add Conversation Memory with Message History

# Create an empty dictionary to store chat histories for each session.
store = {}

# Define a function to get (or create) the chat history for a given session.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # If the session_id does not exist in the store, create a new chat history for it.
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    # Return the chat history for the given session_id.
    return store[session_id]

# Wrap the language model with a mechanism to remember conversation history.
# 'RunnableWithMessageHistory' takes the model and a function to retrieve session history.
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Session 1: Alice
config1 = {"configurable": {"session_id": "session1"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Alice")],
    config=config1,
)
print("Session 1:", response.content)

# Session 2: Bob
config2 = {"configurable": {"session_id": "session2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config2,
)
print("Session 2:", response.content)

# Follow-up in Session 1
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config1,
)
print("Session 1:", response.content) 

# Follow-up in Session 2
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config2,
)
print("Session 2:", response.content)  

#%% 5. Using Prompt Templates

# Add a system message and allow for language customization
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chain prompt and model
chain = prompt | model

# Example: Respond in Spanish
response = chain.invoke(
    {"messages": [HumanMessage(content="Hi! I'm Bob")], "language": "Spanish"}
)
print("Spanish response:", response.content)

#%% 6. Wrap Complex Chain in Message History

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "session3"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="Hi! I'm Jim")], "language": "English"},
    config=config,
)
print("Personalized greeting:", response.content)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="What's my name?")], "language": "English"},
    config=config,
)
print("Remembered name:", response.content)

#%% 7. Managing Conversation History (Trimming)

# Prevent the message history from growing too large for the model's context window

# Define a trimmer to keep only the last N tokens/messages
trimmer = trim_messages(
    max_tokens=40, # Limits the total number of tokens in the trimmed chat history to 40.
    strategy="last", # Keeps the most recent messages, trimming from the start to stay within the token limit.
    token_counter=model, # Uses the model's internal method to count tokens, ensuring compatibility with the model's context window.
    include_system=True, # Keep initial SystemMessage, as it often contains important instructions.
    allow_partial=False, # Only includes whole messages; if a message would exceed the limit, it is omitted rather than partially included.
    start_on="human", # Trimmed history starts with a HumanMessage (or a SystemMessage followed by a HumanMessage), which is expected by most chat models.
)

# Dummy conversation
messages = [
    SystemMessage(content="You're a good assistant"),
    HumanMessage(content="Hi! I'm Bob"),
    AIMessage(content="Hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="Nice"),
    HumanMessage(content="What's 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="Thanks"),
    AIMessage(content="No problem!"),
    HumanMessage(content="Having fun?"),
    AIMessage(content="Yes!"),
]

# Apply trimming
trimmed_messages = trimmer.invoke(messages)
print("Trimmed messages:", trimmed_messages)

# Use trimmer in the chain
chain_with_trimmer = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain_with_trimmer.invoke(
    {
        "messages": messages + [HumanMessage(content="What's my name?")],
        "language": "English",
    }
)
print("After trimming (name):", response.content)

response = chain_with_trimmer.invoke(
    {
        "messages": messages + [HumanMessage(content="What math problem did I ask?")],
        "language": "English",
    }
)
print("After trimming (math):", response.content)

#%% 8. Streaming Responses

# Stream the model's response token by token for better UX
config = {"configurable": {"session_id": "session4"}}
print("Streaming response: ", end="")
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="Hi! I'm Todd. Tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
print()

