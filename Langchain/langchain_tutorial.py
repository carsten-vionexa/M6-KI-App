# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:22:00 2025

@author: milos
"""

# pip install langchain langchain_ollama langchain_openai


#%% Simple QA

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Create model (local Ollama)
model = ChatOllama(model="llama3.1")

messages = [
    # SystemMessage: Represents a system-level instruction or context for the model.
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"), # Represents a message from the user.
]

model.invoke(messages)

parser = StrOutputParser()
"""
Extracts the Text Response:
Language models in LangChain return a complex object (such as AIMessage) that includes 
the generated text and metadata like token usage, model name, and other details.

StrOutputParser extracts just the main string content (the actual response text).
"""

result = model.invoke(messages)
parser.invoke(result)


#%% Chain (Kette)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Create prompt template
# System message is an instruction for the language model, telling it what task to perform.
system_template = "You are a translator. Translate ONLY the following text into {language}, and do not add anything else:"

# Combine multiple messages (from system, user, or assistant) into a single prompt object.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template), # The first message is a system message using system_template.
    ("user", "{text}") # The second message is a user message.
])

"""
# System messages provide instructions; user messages provide the content.
The resulting prompt (as a list of messages) is:

System: "Translate the following into italian:"
User: "hi"
"""

# 2. Create model (local Ollama)
model = ChatOllama(model="llama3.1")

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain - each module’s output becomes the next module’s input, forming a pipeline.
chain = prompt_template | model | parser

# 5. Example usage
inputs = {"language": "italian", "text": "bye"}
result = chain.invoke(inputs)
print(result)
