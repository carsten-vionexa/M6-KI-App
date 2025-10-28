# Schritt 1: Projekteinrichtung
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

# Schritt 2: Laden des Datensatzes
dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)

# Split the dataset into training and testing sets
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Schritt 3: Auswahl von Modell und Tokenizer
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # positive, negative, neutral

# Schritt 4: Datenvorverarbeitung (Tokenisierung)


# Schritt 5: Fine-Tuning des Modells


# Schritt 6: Evaluierung des Modells

