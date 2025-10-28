# src/utils.py
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def load_financial_phrasebank(split_ratio=0.8, seed=42):
    """Lädt und splittet den Financial PhraseBank Datensatz."""
    dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
    dataset = dataset["train"].train_test_split(test_size=1 - split_ratio, seed=seed)
    return dataset

def compute_metrics(pred):
    """Berechnet Accuracy und Macro-F1 für HuggingFace Trainer."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro")
    rec = recall_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1, "precision": prec, "recall": rec}