from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def create_chat_chain(model_name="llama3.1"):
    llm = Ollama(model=model_name)
    prompt = ChatPromptTemplate.from_template(
        """Dies ist ein Gespräch über Seminare und Trainings.
Nutze den bisherigen Verlauf, um Folgefragen zu verstehen.
Wenn du etwas nicht weißt, sag "Ich weiß es nicht".

Verlauf:
{history}

Neue Frage:
{question}

Antwort:"""
    )
    chain = prompt | llm
    def run_with_history(question, history_text):
        return chain.invoke({"history": history_text, "question": question})
    return run_with_history