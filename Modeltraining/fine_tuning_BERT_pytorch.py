# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 20:36:43 2025

@author: milos
"""

"""
Tutorial: Fine-Tuning BERT for Sequence Classification with a Custom Training Loop 

In this tutorial, we will fine-tune a BERT model on the Microsoft Research 
Paraphrase Corpus (MRPC) dataset using a custom training loop implemented 
from scratch with PyTorch. 

Key Features
Size: The dataset contains 5,801 pairs of sentences.
Source: Sentences are extracted from newswire articles and online news sources.
Annotation: Each pair is labeled by human annotators as either a paraphrase 
(semantically equivalent) or not a paraphrase.
Purpose: Designed to support research in paraphrase detection, semantic similarity, 
and related NLP tasks.


We will cover:
- Loading and preprocessing the dataset
- Tokenization and dynamic padding
- Preparing PyTorch DataLoaders
- Setting up the model, optimizer, and learning rate scheduler
- Implementing a training loop with GPU support
- Implementing an evaluation loop with metric tracking

Prerequisites:
- Install required libraries: transformers, datasets, torch, evaluate, tqdm
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate

# Step 1: Load the MRPC dataset from the Hugging Face Hub
raw_datasets = load_dataset("glue", "mrpc")

examples = raw_datasets['train'].select(range(2))
for example in examples:
    print(example)

# Step 2: Load the tokenizer for the pretrained BERT model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Step 3: Tokenize the dataset (sentence pairs) with truncation
def tokenize_function(example):
    """
    Takes an example (containing two sentences) and applies the tokenizer to both, 
    with truncation enabled (ensuring sequences do not exceed the model’s maximum input length).
    Result: tokenized_datasets contains the same splits as raw_datasets, 
    but each example now includes fields like input_ids, attention_mask, and 
    token_type_ids—all required for BERT.
    """
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

examples = tokenized_datasets['train'].select(range(2))
for example in examples:
    print(example)


# Step 4: Postprocess the tokenized datasets to prepare for PyTorch
# Remove columns not expected by the model, rename 'label' to 'labels', and set format to 
# PyTorch tensors
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

examples = tokenized_datasets['train'].select(range(2))
for example in examples:
    print(example)


# Step 5: Create DataCollator for dynamic padding within batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# The DataCollatorWithPadding automatically pads each batch of tokenized sequences to 
# the length of the longest sequence in that batch.
# Padding to the longest sequence in the entire dataset wastes memory and computation.
# Dynamic (batch-level) padding is more efficient: each batch is only as long as its longest member.


# Step 6: Create PyTorch DataLoaders for training and evaluation
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator
)

# Step 7: Load the pretrained BERT model with a classification head (2 labels for MRPC)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Step 8: Setup device for GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Step 9: Initialize the optimizer (AdamW) and learning rate scheduler (linear decay)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# model.parameters(): All trainable parameters of your model are passed to the optimizer.
# lr=5e-5: Sets the learning rate, which controls how much the model weights are updated at each step.
# weight_decay=0.01: Adds L2 regularization to help prevent overfitting by penalizing large weights.

num_epochs = 3 # Number of times the model will see the entire training dataset.
num_training_steps = num_epochs * len(train_dataloader) # Total updates = epochs × batches per epoch.

# Learning rate scheduler that linearly decreases the learning rate from its initial value
# to zero over the course of training.
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Step 10: Prepare metric for evaluation (accuracy and F1 score)
metric = evaluate.load("glue", "mrpc")
# Loads the official GLUE evaluation metrics for the MRPC task using the evaluate library.
# Using the official GLUE metrics ensures your results are directly comparable to other models and benchmarks in the research community.

# Step 11: Training loop with progress bar and GPU support
model.train()
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move batch to device (GPU or CPU)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass to compute outputs and loss
        # Feeds the batch into the model to get predictions.
        # Calculates how far off the predictions are from the correct answers.
        outputs = model(**batch)
        loss = outputs.loss

        # Backpropagation - how each model parameter contributed to the loss.
        loss.backward()

        # Update parameters and learning rate to reduce the loss.
        optimizer.step() 
        lr_scheduler.step()
        optimizer.zero_grad() # Clears old gradients to get ready for the next batch.

        # Update progress bar
        progress_bar.update(1)

# Step 12: Evaluation loop to compute metrics on validation set
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Accumulate predictions and references for metric calculation
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Compute final evaluation metrics
eval_results = metric.compute()
print(f"Validation Accuracy: {eval_results['accuracy']:.4f}")
print(f"Validation F1 Score: {eval_results['f1']:.4f}")

# Optional: Save the fine-tuned model and tokenizer
# model.save_pretrained("./mrpc-bert-finetuned-custom")
# tokenizer.save_pretrained("./mrpc-bert-finetuned-custom")

"""
Summary:
- This script demonstrates how to fine-tune BERT on MRPC without the Trainer API.
- It uses PyTorch DataLoaders, a custom training loop, and evaluation loop.
- GPU support is enabled automatically if CUDA is available.
- Dynamic padding is handled by DataCollatorWithPadding for efficiency.
- The AdamW optimizer and a linear scheduler are used to replicate Trainer defaults.
- Metrics include accuracy and F1 score, standard for MRPC evaluation.
"""
