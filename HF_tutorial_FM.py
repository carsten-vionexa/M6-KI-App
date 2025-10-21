# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:12:27 2025

@author: milos
"""

# Hugging Face Transformers

#############################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread("full_nlp_pipeline.png")  
plt.imshow(img)
plt.axis("off")
plt.show()
#############################################

#############################################
img = mpimg.imread("full_model.png")  
plt.imshow(img)
plt.axis("off")
plt.show()
#############################################


#%%  Install Required Libraries

# pip install transformers datasets

#%% Walkthrough

# ==========================================
# 1. Import Required Libraries
# ==========================================

from transformers import AutoModel, AutoTokenizer, BertModel
import torch

# ==========================================
# 2. Loading a Pretrained Model
# ==========================================

# AutoModel automatically selects the right architecture for the checkpoint.
model = AutoModel.from_pretrained("bert-base-cased")

# The AutoModel class is a simple wrapper designed to fetch the appropriate model 
# architecture for a given checkpoint. 
# It will guess the appropriate model architecture and instantiate the 
# correct model class. 
# BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) 

# ==========================================
# 3. Saving and Reloading the Model
# ==========================================
# Models can be saved locally and reloaded later.
save_directory = "my_bert_model"
model.save_pretrained(save_directory)

# To reload the model from the saved directory:
model_reloaded = AutoModel.from_pretrained(save_directory)

# ==========================================
# 4. Uploading Model to Hugging Face Hub
# ==========================================
# To share your model, you can push it to the Hugging Face Hub.
# You need to be logged in (see documentation for notebook_login or CLI login).

# from huggingface_hub import login
# # Replace 'hf_xxx...' with your actual token
# login(token="hf")
# model.push_to_hub("my-awesome-model")

# ==========================================
# 5. Tokenizing Text
# ==========================================
# Tokenizers convert text into the numerical format required by models.

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Encode a single sentence
sentence = "Hello, I'm a single sentence!"
encoded_input = tokenizer(sentence)
print("Encoded input:", encoded_input)
# input_ids: The numerical representation of each token in sentence.
# attention_mask: Indicates which tokens are actual words (1) and which are padding (0).
# The attention mask helps the model ignore padding tokens during computation.

# Decoding back to text (shows special tokens)
decoded_text = tokenizer.decode(encoded_input["input_ids"])
print(decoded_text)
# Special tokens are added to better represent the sentence boundaries, 
# such as the beginning of a sentence ([CLS]) or separator between sentences ([SEP]). 
# They are primarily used when a model was pretrained with them (automatically added) 

# ==========================================
# 6. Tokenizing Multiple Sentences
# ==========================================

sentences = ["How are you?", "I'm fine, thank you!"]
encoded_batch = tokenizer(sentences, padding=True, return_tensors="pt")
print(encoded_batch)

# ==========================================
# 7. Padding and Truncation
# ==========================================
# padding=True: Ensures all sequences in the batch are the same length by adding padding tokens 
# to shorter sentences. Models require batches of data to be the same size. 
# truncation=True: Cuts off sequences that are too long for the model’s maximum input length.
# return_tensors="pt": Returns the output as PyTorch tensors, which are needed for model inference.


encoded_padded_truncated = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt"
)
print(encoded_padded_truncated)


#%% Transformer pipeline

# https://huggingface.co/learn/llm-course/chapter1/3?fw=pt

#%% Load Pretrained Model for Sentiment Analysis

from transformers import pipeline

# Load sentiment analysis pipeline.
# It connects a model with its necessary preprocessing and postprocessing steps, 
# allowing us to directly input any text and get an intelligible answer.
# loads a pretrained model and tokenizer designed for sentiment analysis
# It uses a default model (distilbert-base-uncased-finetuned-sst-2-english), 
# which is trained to classify text as either "POSITIVE" or "NEGATIVE".
classifier = pipeline("sentiment-analysis")

# available tasks are ['audio-classification', 'automatic-speech-recognition', 
# 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 
# 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-text-to-text', 
# 'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 
# 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 
# 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 
# 'text2text-generation', 'token-classification', 'translation', 'video-classification', 
# 'visual-question-answering', 'vqa', 'zero-shot-audio-classification', 
# 'zero-shot-classification', 'zero-shot-image-classification', 
# 'zero-shot-object-detection', 'translation_XX_to_YY']"

# Try it on some text
result = classifier("I love using Hugging Face models!")
print(result)

# List of texts
texts = ["I hate this!", "This is great!", "It's okay, not bad."]
results = classifier(texts)

for text, res in zip(texts, results):
    print(f"Text: {text}\nLabel: {res['label']}, Score: {res['score']:.4f}\n")
    
#%% Text generation

from transformers import pipeline

# Initialize the text generation pipeline with a specific model (optional)
generator = pipeline("text-generation")  

# Prompt for completion
prompt = "The cat sat on the mat and then"

# Generate continuation
generated = generator(
    prompt,
    max_new_tokens=40,        # total length including prompt
    num_return_sequences=1,
    truncation=True,
    do_sample=True,       # enables randomness
    temperature=0.9,     # controls creativity
    top_k=50,             # filters top K tokens
    top_p=0.95,           # nucleus sampling
    pad_token_id=50256    # for GPT-2 to avoid warnings; GPT-2’s special token
)

# Print the completed text
print(generated[0]['generated_text'])

#%% Text Summarization

# This code demonstrates text summarization with BERT by leveraging a pre-trained T5 model 
# (fine-tuned on summarization tasks). It tokenizes the article, generates a summary using the model, 
# decodes the generated summary tokens, and prints both the original article and the summarized text.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load T5 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Define the article to summarize
article = """Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that deals with the interaction between computers and human language. In particular, NLP focuses on the branch of computer science concerned with the interaction between computers and human (natural) languages. NLP applications are able to analyze large amounts of natural language data to extract information, derive insights, and generate reports."""

# Add task prefix for T5
input_text = "summarize: " + article

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], 
                             max_length=50, 
                             min_length=20, 
                             length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print results
print("Original Article:\n", article)
print("\nSummarized Text:\n", summary)

#%% Question Answering with BERT

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load a model fine-tuned for question answering
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Input question and context
question = "What is the capital of France?"
context = "France is a country located in Western Europe. The capital of France is Paris."

# Tokenize question + context together
inputs = tokenizer(question, context, return_tensors="pt")

# Run model
outputs = model(**inputs)

# Get start and end positions
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)

# Convert tokens to answer text
answer_ids = inputs["input_ids"][0][start_index : end_index + 1]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

print("Answer:", answer)


    
    
#%% Hugging Face Tokenizer and Model Pipeline

# ==========================================
# 1. Import Required Libraries
# ==========================================
# We'll use transformers for the tokenizer and model, and torch for tensor handling.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==========================================
# 2. Load Pretrained Tokenizer and Model
# ==========================================
# We'll use a DistilBERT model fine-tuned on sentiment analysis (SST-2 dataset).

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# ==========================================
# 3. Prepare Input Sequences
# ==========================================
# You can process single or multiple sentences at once.
# The tokenizer will handle tokenization, conversion to IDs, padding, truncation, 
# and creation of attention masks.

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]

# ==========================================
# 4. Tokenize Input Sequences
# ==========================================
# The tokenizer returns a dictionary with:
# - input_ids: token IDs (with special tokens like [CLS] and [SEP])
# - attention_mask: indicates which tokens are actual words vs. padding
# Setting padding=True pads all sequences to the same length.
# Setting truncation=True ensures sequences don't exceed the model's max length.
# return_tensors="pt" returns PyTorch tensors.

tokens = tokenizer(
    sequences,
    padding=True,        # Pad to the longest sequence in the batch
    truncation=True,     # Truncate sequences longer than the model's max length
    return_tensors="pt"  # Return PyTorch tensors
)
# print(tokenizer.model_max_length)

print("Tokenized inputs:")
print(tokens)

# ==========================================
# 5. Inspect Special Tokens
# ==========================================
# The tokenizer automatically adds special tokens required by the model.
# Let's see the input IDs and decode them back to text.

print("\nInput IDs for first sequence:", tokens["input_ids"][0])
print("Decoded:", tokenizer.decode(tokens["input_ids"][0]))

# ==========================================
# 6. Pass Inputs to the Model
# ==========================================
# The model expects the same fields as returned by the tokenizer.
# We'll perform inference without computing gradients.

with torch.no_grad():
    outputs = model(**tokens)

print("\nModel outputs (logits):")
print(outputs.logits)
# Logits: Rohwerte, noch keine Wahrscheinlichkeiten, beliebiger Zahlenbereich

# ==========================================
# 7. Interpreting the Output
# ==========================================
# For sequence classification, the output is usually logits for each class.
# To get probabilities, apply softmax (jeder Wert zwischen 0 und 1 und alle Werte 
# zusammen genau 1 ergeben - Wahrscheinlichkeitsverteilung

probs = torch.nn.functional.softmax(outputs.logits, dim=1)
print("\nProbabilities for each class:")
print(probs)
model.config.id2label
# First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
# Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005


# ==========================================
# 8. Summary of Key Concepts
# ==========================================
# - The tokenizer handles all preprocessing: tokenization, IDs, padding, truncation, attention masks.
# - Special tokens are added automatically as required for each model.
# - The output from the tokenizer can be directly fed into the model.
# - The model outputs logits, which can be converted to probabilities.
# - This workflow supports both single and batch inputs, and can return tensors for PyTorch or NumPy.

