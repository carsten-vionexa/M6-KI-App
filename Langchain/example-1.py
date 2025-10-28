from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model.invoke(messages)

parser = StrOutputParser()
result = model.invoke(messages)
parser.invoke(result)

#Prompt-Template
from langchain_core.prompts import ChatPromptTemplate

system_template = "You are a translator. Translate ONLY the following text into {language}, and do not add anything else:"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

model = ChatOllama(model="llama3.1")
parser = StrOutputParser()

chain = prompt_template | model | parser

inputs = {"language": "italian", "text": "bye"}
result = chain.invoke(inputs)
print(result)