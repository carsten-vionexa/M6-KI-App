# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:10:46 2025

@author: milos
"""

# Ollama tutorial

# pip install ollama PyPDF2

import ollama
import PyPDF2       # Library for reading PDF files

#%% Summarization

prompt = (
    "Summarize the following text: "
    "Artificial intelligence (AI) has rapidly transformed various industries over the past decade. "
    "From healthcare and finance to transportation and entertainment, AI technologies are enabling automation, "
    "improving decision-making, and creating new opportunities for innovation. As organizations continue to adopt "
    "AI-driven solutions, it becomes increasingly important to consider the ethical implications and ensure responsible use. "
    "This includes addressing concerns such as data privacy, algorithmic bias, and the potential impact on employment. "
    "By fostering transparency and collaboration among stakeholders, society can maximize the benefits of AI while mitigating its risks."
)

response = ollama.generate(model="llama3.1", prompt=prompt)
print(response['response'])

#%% PDF summarization

def extract_text_from_pdf(file_path):
    """
    Extracts all text from a PDF file.
    :param file_path: Path to the PDF file.
    :return: Combined text from all pages.
    """
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)  # Create a PDF reader object
        for page in reader.pages:        # Iterate over each page in the PDF
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add the text of each page
    return text

# Specify the path to your PDF file
pdf_path = "C:\\Users\\milos\\Desktop\\DATA\\Educx\\Modul_6_KI\\KI_Lernplan\\KI_W1_T4\\Ollama_tutorial\\Docker_tutorial.pdf"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Prepare the prompt for the language model
prompt = (
    "Summarize the following text: "
    + pdf_text
)

# Generate the summary using Ollama
response = ollama.generate(
    model="llama3.1",           # Specify the model to use
    prompt=prompt,              # Provide the prompt with extracted text
)

# Print the generated summary
print(response['response'])


#%% Chat - no memory

model = 'llama3.1:latest'  # Specify the model to use

while True:
    user_input = input("You: ")  # Prompt the user for input
    # Allow the user to exit the chat loop
    if user_input.lower() in ['exit', 'quit', 'q']:
        break
    # Send only the current user input to the model (no memory of previous turns)
    response = ollama.generate(
        model=model,                # Model to use for generation
        prompt=user_input,          # The user's current message as the prompt
        options={
            "temperature": 0.1,     # Controls randomness/creativity of output
            "num_predict": 100,     # Maximum number of tokens to generate in the response
            "top_p": 0.8, 
        }
    )
    # Print the model's response, removing leading/trailing whitespace
    print("Assistant:", response['response'].strip())
    
"""
Temperature 
Niedrige Werte: Das Modell wählt fast immer die wahrscheinlichste (vorhersehbare) Antwort - 
    sachliche, zuverlässige Antworten.
Hohe Werte: Das Modell wählt auch weniger wahrscheinliche Wörter - 
    kreative und abwechslungsreiche Antworten.
    
top_p (Nucleus Sampling)
Das Modell berechnet für alle möglichen Wörter/Tokens eine Wahrscheinlichkeit.
Bei Nucleus Sampling werden die wahrscheinlichsten Wörter so lange aufaddiert, 
bis ihre Gesamtwahrscheinlichkeit den Wert von top_p erreicht.

Aus diesem „Nucleus“ (Kern) an Wörtern wird dann zufällig das nächste Wort ausgewählt.

„Katze“: 0.5
„Hund“: 0.3
„Vogel“: 0.1
„Fisch“: 0.05
„Maus“: 0.05

Wenn top_p = 0.8, werden „Katze“ und „Hund“ (0.5 + 0.3 = 0.8) in die Auswahl genommen. 
Die anderen Wörter werden ignoriert.

Niedriges top_p (z.B. 0.2): Nur die allerwahrscheinlichsten Wörter werden ausgewählt → 
    sehr vorhersehbare, fokussierte Antworten.

Hohes top_p (z.B. 0.95): Mehr Wörter stehen zur Auswahl → 
    die Antworten werden kreativer und vielfältiger.

top_k
Hier schaut das Modell bei jedem Schritt auf die top_k wahrscheinlichsten Wörter und 
wählt daraus zufällig das nächste Wort.

Kleines top_k (z.B. 1): Das Modell wählt immer das wahrscheinlichste Wort → 
    deterministische, oft monotone Antworten.

Größeres top_k (z.B. 40): Das Modell kann auch weniger wahrscheinliche Wörter auswählen → 
    mehr Variation und Kreativität.

top_k = 40, top_p = 0.9
Es werden maximal 40 Tokens betrachtet.
Von diesen werden nur die Tokens gewählt, deren kumulierte Wahrscheinlichkeit mindestens 90 % beträgt.
"""
   

#%% Chat with memory

model = 'llama3.1'  
messages = []       # Initialize an empty list to store the conversation history

def chat(message):
    # Add the user's message to the conversation history
    messages.append({'role': 'user', 'content': message})
    
    # Send the full conversation history to the model and get its response
    response = ollama.chat(model=model, messages=messages)
    
    # Add the assistant's response to the conversation history
    # This ensures future responses from the model can reference both user and assistant messages.
    messages.append({'role': 'assistant', 
                     'content': response['message']['content']})
    
    # Print the assistant's reply to the console
    print("Assistant:", response['message']['content'])

# Start a loop to continuously chat with the user
while True:
    user_input = input("You: ")  # Prompt the user for input
    # Exit the chat loop if the user types 'exit', 'quit', or 'q'
    if user_input.lower() in ['exit', 'quit', 'q']:
        break
    # Call the chat function with the user's input
    chat(user_input)

#%% Cloud models

from ollama import Client

client = Client()

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)


#%% Models via API - Direct client-to-cloud communication without local proxy

from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'a4fc5cf0365a4845a78ae104de378d11.ulBsrl5dqaSHX21s_Ke5XMiZ'}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
