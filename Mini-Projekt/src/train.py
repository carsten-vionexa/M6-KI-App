# src/train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import DatasetDict
from utils import load_financial_phrasebank, compute_metrics
from transformers import set_seed

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def main():
    import os
    os.makedirs("./reports/figs", exist_ok=True)
    
    # 1. Daten laden
    dataset = load_financial_phrasebank(split_ratio=0.8, seed=42)
    print(dataset)

    # 2. Tokenizer laden
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Tokenisierung anwenden
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 4. Modell laden
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # positiv / neutral / negativ
    )

    # 5. TrainingArguments definieren
    set_seed(42)
    training_args = TrainingArguments(
        output_dir="./reports/models",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./reports/logs",
        logging_steps=50,
        report_to="none"  # kein WandB/Hub-Logging
    )

    # 6. Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. Training starten
    trainer.train()

    # 8. Evaluation auf Testset
    eval_results = trainer.evaluate()
    print("\n--- Evaluation Results ---")
    for k, v in eval_results.items():
        print(f"{k}: {v:.4f}")



    # Kontrollausgabe
    print(tokenized_datasets["train"][0])

    #Confusionsmatrix
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    preds_output = trainer.predict(tokenized_datasets["test"])
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix – Financial PhraseBank")
    plt.tight_layout()
    plt.savefig("./reports/figs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()