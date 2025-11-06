# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:58:21 2025

@author: milos
"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# Setup LLM
llm = ChatOllama(model='gpt-oss:120b-cloud')

# Setup search tool
search = TavilySearchResults(
    k=3,
    tavily_api_key=""
)
tools = [search]

# Define simple prompt template only with 'messages' placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an agent that can use tools to answer questions."),
        ("placeholder", "{messages}"),
    ]
)

# Create React agent
agent = create_agent(model=llm, 
                     tools=tools, 
                     system_prompt="You are a helpful assistant. Be concise and accurate.")

# User query
query = "What is the weather in Luxembourg today?"

# Prepare input message structure
input_messages = {"messages": [("human", query)]}

# Invoke agent
response = agent.invoke(input_messages)

for message in response["messages"]:
    # Determine role by message class type
    if message.__class__.__name__ == "HumanMessage":
        role = "human"
    elif message.__class__.__name__ == "AIMessage":
        role = "assistant"
    elif message.__class__.__name__ == "ToolMessage":
        role = "agent"
    else:
        role = "unknown"
    
    content = getattr(message, "content", "")
    
    if role == "agent":
        print(f"[Agent action]: {content}\n")
    elif role == "human":
        print(f"[User]: {content}\n")
    elif role == "assistant":
        print(f"[Assistant final answer]:\n{content}\n")
        

